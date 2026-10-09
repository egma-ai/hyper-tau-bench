"""The search_corpus experiment tool: reading the kit, judging, reporting."""

import base64
import io
import json
import zipfile
from email.message import EmailMessage

import pytest

from tau2.hyper.harnesses.codex import CORPUS_SEARCH_SKILL_PATH, CodexSandboxBuilder
from tau2.hyper.harnesses.factory import create_developer_builder
from tau2.hyper.sandbox import callback_mcp
from tau2.hyper.sandbox.callback_broker import (
    CallbackBroker,
    CallbackBrokerError,
    CallbackQuotaError,
)
from tau2.hyper.sandbox.corpus_search import (
    SECTION_CHARS,
    TOOL_DESCRIPTION,
    CorpusSearch,
    material_files,
    read_units,
)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _docx(text: str) -> bytes:
    body = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )
    return _zip({"word/document.xml": body.encode()})


def _xlsx(cells: list[str]) -> bytes:
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    shared = "".join(f"<si><t>{c}</t></si>" for c in cells)
    row = "".join(f'<c t="s"><v>{i}</v></c>' for i in range(len(cells)))
    return _zip(
        {
            "xl/sharedStrings.xml": f"<sst {ns}>{shared}</sst>".encode(),
            "xl/workbook.xml": f'<workbook {ns}><sheets><sheet name="Fees"/></sheets></workbook>'.encode(),
            "xl/worksheets/sheet1.xml": f"<worksheet {ns}><sheetData><row>{row}</row></sheetData></worksheet>".encode(),
        }
    )


def _email() -> bytes:
    message = EmailMessage()
    message["From"] = "ops@bank.example"
    message["Subject"] = "Overdraft fee waivers"
    message.set_content("Agents may waive one overdraft fee per year.")
    message.add_attachment(PNG, maintype="image", subtype="png", filename="card.png")
    return bytes(message)


@pytest.fixture
def kit(tmp_path):
    (tmp_path / "uploaded_materials" / "mail").mkdir(parents=True)
    (tmp_path / "framework").mkdir()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "simulations").mkdir()
    (tmp_path / "sop.md").write_text("# SOP\nRefunds need a receipt.\n")
    long_lines = "".join(f"line {i} about baggage allowances\n" for i in range(2000))
    (tmp_path / "uploaded_materials" / "handbook.md").write_text(long_lines)
    (tmp_path / "uploaded_materials" / "mail" / "waiver.eml").write_bytes(_email())
    (tmp_path / "uploaded_materials" / "policy.docx").write_bytes(
        _docx("Refund within 30 days")
    )
    (tmp_path / "uploaded_materials" / "fees.xlsx").write_bytes(
        _xlsx(["Wire fee", "25"])
    )
    (tmp_path / "uploaded_materials" / "bundle.zip").write_bytes(
        _zip({"inner/note.txt": b"Refund exceptions for veterans", "clip.mp4": b"\0\0"})
    )
    (tmp_path / "uploaded_materials" / "screenshot.png").write_bytes(PNG)
    (tmp_path / "uploaded_materials" / "walkthrough.mp4").write_bytes(b"\0" * 64)
    (tmp_path / "framework" / "harness.md").write_text("refund refund refund")
    (tmp_path / "workspace" / "agent.py").write_text("# refund")
    (tmp_path / ".hidden.md").write_text("refund")
    return tmp_path


class FakeDecide:
    """Probability 0.9 when the text mentions the keyword, else 0.1."""

    def __init__(self, keyword: str = "refund", fail_on: str | None = None):
        self.keyword = keyword
        self.fail_on = fail_on
        self.calls = []

    def __call__(self, unit, question):
        self.calls.append((unit.label, unit.kind, question))
        if self.fail_on and self.fail_on in unit.label:
            raise RuntimeError("upstream 400")
        if unit.kind == "image":
            assert unit.payload.startswith("data:image/png;base64,")
            return 0.6, 85
        return (0.9 if self.keyword in unit.payload.lower() else 0.1), 100


def test_material_files_exclude_framework_workspace_simulations_and_hidden(kit):
    files = material_files(kit)
    assert "sop.md" in files
    assert "uploaded_materials/mail/waiver.eml" in files
    assert not any(
        f.startswith(("framework/", "workspace/", "simulations/")) for f in files
    )
    assert ".hidden.md" not in files


def test_reads_office_email_zip_images_and_sections_long_files(kit):
    search = CorpusSearch(kit, decide=FakeDecide())
    units = {unit.label: unit for unit in search._load()}
    assert "Refund within 30 days" in units["uploaded_materials/policy.docx"].payload
    assert "Wire fee\t25" in units["uploaded_materials/fees.xlsx"].payload
    email_text = units["uploaded_materials/mail/waiver.eml"].payload
    assert (
        "Subject: Overdraft fee waivers" in email_text
        and "waive one overdraft" in email_text
    )
    assert units["uploaded_materials/mail/waiver.eml > card.png"].kind == "image"
    assert units["uploaded_materials/screenshot.png"].kind == "image"
    assert "veterans" in units["uploaded_materials/bundle.zip > inner/note.txt"].payload
    sections = [u for u in units.values() if u.path == "uploaded_materials/handbook.md"]
    assert len(sections) > 1 and all(len(u.payload) <= SECTION_CHARS for u in sections)
    assert sections[0].label.endswith("(lines 1-" + str(sections[0].lines[1]) + ")")
    assert dict(search._skipped) == {
        "uploaded_materials/walkthrough.mp4": "audio/video",
        "uploaded_materials/bundle.zip > clip.mp4": "audio/video",
    }


def test_search_lists_hits_saves_every_result_and_reports_skips(kit):
    decide = FakeDecide()
    search = CorpusSearch(kit, decide=decide)
    report = search.search("Does this file state a refund rule?")
    lines = report.splitlines()
    assert lines[0].startswith("Asked ")
    assert "Skipped (not read): 2 audio/video." in report
    hits = [line for line in lines if line[:4] in ("0.90", "0.60")]
    assert any("sop.md" in line for line in hits)
    assert any("policy.docx" in line for line in hits)
    assert any("bundle.zip > inner/note.txt" in line for line in hits)
    assert not any("handbook.md" in line for line in hits)
    saved = sorted((kit / "corpus_search").glob("*.json"))
    assert len(saved) == 1
    data = json.loads(saved[0].read_text())
    assert data["question"] == "Does this file state a refund rule?"
    assert len(data["results"]) == len(search._load())
    assert data["results"][0]["probability"] >= data["results"][-1]["probability"]
    assert search.usage()["decisions"] == len(search._load())
    assert search.usage()["cost_usd"] > 0


def test_repeat_questions_are_cached_and_quota_is_enforced(kit):
    decide = FakeDecide()
    search = CorpusSearch(kit, decide=decide, max_calls=2)
    search.search("Q?")
    first = len(decide.calls)
    search.search("Q?")
    assert len(decide.calls) == first
    with pytest.raises(Exception, match="limit of 2 calls"):
        search.search("Another?")


def test_path_filter_threshold_and_errors(kit):
    search = CorpusSearch(kit, decide=FakeDecide(fail_on="fees.xlsx"))
    report = search.search(
        "Q?", min_probability=0.95, path="/workspace/uploaded_materials/"
    )
    assert "sop.md" not in report
    assert "Items with probability >= 0.95: 0" in report
    assert "Could not be judged (API errors): 1." in report
    with pytest.raises(Exception, match="No readable task-material files"):
        search.search("Q?", path="nowhere")


def test_corpus_is_fixed_before_the_developer_writes_notes(kit):
    search = CorpusSearch(kit, decide=FakeDecide())
    (kit / "uploaded_materials" / "my_notes.md").write_text("refund summary")
    search.search("Q?")
    assert not any("my_notes.md" in label for label, _, _ in search._decide.calls)


def test_read_units_skips_binary_and_unknown_pdf_reader_gracefully():
    assert read_units("blob.bin", "blob.bin", b"\0\1\2" * 100) == (
        [],
        [("blob.bin", "binary")],
    )
    units, skipped = read_units("empty.md", "empty.md", b"   ")
    assert units == [] and skipped == [("empty.md", "no text")]


def test_broker_dispatches_search_corpus_and_maps_quota(kit):
    search = CorpusSearch(kit, decide=FakeDecide(), max_calls=1)
    broker = CallbackBroker(kit, toolkit=object(), corpus_search=search)
    assert broker.corpus_search_tool_enabled
    report = broker.dispatch(
        token=broker.token, tool="search_corpus", arguments={"question": "Q?"}
    )
    assert report.startswith("Asked ")
    with pytest.raises(CallbackQuotaError):
        broker.dispatch(
            token=broker.token, tool="search_corpus", arguments={"question": "Q?2"}
        )
    assert broker.metadata()["corpus_search"]["calls_used"] == 1
    broker.close()


def test_broker_without_corpus_search_rejects_the_tool(kit):
    broker = CallbackBroker(kit, toolkit=object())
    assert not broker.corpus_search_tool_enabled
    assert "corpus_search" not in broker.metadata()
    with pytest.raises(CallbackBrokerError, match="not available"):
        broker.dispatch(
            token=broker.token, tool="search_corpus", arguments={"question": "Q?"}
        )
    broker.close()


def test_mcp_lists_search_corpus_only_when_enabled(monkeypatch):
    monkeypatch.delenv("TAU2_CORPUS_SEARCH_TOOL_ENABLED", raising=False)
    listed = callback_mcp._handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert "search_corpus" not in [t["name"] for t in listed["result"]["tools"]]
    monkeypatch.setenv("TAU2_CORPUS_SEARCH_TOOL_ENABLED", "1")
    listed = callback_mcp._handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = {t["name"]: t for t in listed["result"]["tools"]}
    assert tools["search_corpus"]["description"] == TOOL_DESCRIPTION
    assert tools["search_corpus"]["inputSchema"]["required"] == ["question"]
    assert list(tools)[-1] == "submit"


def test_codex_gets_tool_and_skill_only_in_the_experiment():
    baseline = CodexSandboxBuilder(llm="gpt-6.1-sol")
    files = baseline.runtime_files(include_client_tool=True)
    assert list(files) == [baseline.runtime_config_path]
    assert "search_corpus" not in files[baseline.runtime_config_path]
    assert "extra_tools" not in baseline.harness_config_metadata()

    expert = CodexSandboxBuilder(llm="gpt-6.1-sol", corpus_search=True)
    files = expert.runtime_files(
        include_client_tool=True, include_corpus_search_tool=True
    )
    config = files[expert.runtime_config_path]
    assert '"search_corpus", "submit"]' in config
    assert '"TAU2_CORPUS_SEARCH_TOOL_ENABLED"' in config
    skill = files[CORPUS_SEARCH_SKILL_PATH]
    assert skill.startswith("---\nname: search-corpus\ndescription: ")
    assert expert.harness_config_metadata()["extra_tools"] == ["search_corpus"]


def test_runtime_environment_flags_the_tool_only_when_enabled(kit):
    builder = CodexSandboxBuilder(llm="gpt-6.1-sol")
    plain = CallbackBroker(kit, toolkit=object())
    with_search = CallbackBroker(
        kit, toolkit=object(), corpus_search=CorpusSearch(kit, decide=FakeDecide())
    )
    assert "TAU2_CORPUS_SEARCH_TOOL_ENABLED" not in builder.runtime_environment(plain)
    assert (
        builder.runtime_environment(with_search)["TAU2_CORPUS_SEARCH_TOOL_ENABLED"]
        == "1"
    )
    plain.close()
    with_search.close()


def test_factory_wires_corpus_search_for_codex_only():
    builder = create_developer_builder(
        "codex", "gpt-6.1-sol", None, "max", developer_corpus_search=True
    )
    assert builder.corpus_search_enabled
    with pytest.raises(ValueError, match="codex harness"):
        create_developer_builder(
            "claude-code", "claude-opus-5", None, None, developer_corpus_search=True
        )
