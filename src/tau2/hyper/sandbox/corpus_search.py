"""``search_corpus``: ask one yes/no question of every file in the task materials.

An optional Developer tool for experiments (``--developer-corpus-search``).
It runs in the host process, like the other callback tools, because the
construction container has no network. Every file the kit ships as task
material, or each section of a long file, goes to the OpenAI Decisions API,
which returns the probability that the answer is yes for that file. The tool
lists the files at or above a threshold and saves the result for every file
under ``corpus_search/`` in the kit.

Audio and video are not read. Text formats, Word/Excel/PowerPoint files,
PDFs (when ``pypdf`` is importable), emails and HTML are read as text, images
as images, and members of ``.zip`` archives and email attachments in turn.
"""

from __future__ import annotations

import base64
import email
import email.policy
import hashlib
import html
import io
import json
import os
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from xml.etree import ElementTree

from tau2.hyper.sandbox.callback_mcp import CORPUS_SEARCH_TOOL

DECISIONS_MODEL = "gpt-6-luna"
DECISIONS_PATH = "/v1/decisions"
# Public-beta list price: input tokens only, no output charge.
USD_PER_INPUT_TOKEN = 0.10 / 1_000_000
MAX_CALLS = 200
# Kit entries that are not task material: the harness framework, the
# Developer's own code, test output, and this tool's results.
EXCLUDED_ENTRIES = frozenset({"framework", "workspace", "simulations", "corpus_search"})
SECTION_CHARS = 20_000
MAX_LISTED = 200
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_NESTING = 2

TEXT_SUFFIXES = frozenset(
    ".md .markdown .txt .text .csv .tsv .json .jsonl .yaml .yml .xml .vtt .srt "
    ".log .ini .toml .cfg .rst .py .sql .ics .svg".split()
)
HTML_SUFFIXES = frozenset({".html", ".htm"})
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
CONVERTED_IMAGE_SUFFIXES = frozenset({".bmp", ".tif", ".tiff"})
MEDIA_SUFFIXES = frozenset(
    ".mp4 .mov .m4v .avi .mkv .webm .mp3 .wav .m4a .aac .ogg .oga .flac .opus "
    ".wma .amr".split()
)

# What the Developer sees, defined in the in-container MCP stub.
TOOL_NAME = CORPUS_SEARCH_TOOL["name"]
TOOL_DESCRIPTION = CORPUS_SEARCH_TOOL["description"]
TOOL_INPUT_SCHEMA = CORPUS_SEARCH_TOOL["inputSchema"]


class CorpusSearchError(RuntimeError):
    """A search request that cannot be served."""


class CorpusSearchQuotaError(CorpusSearchError):
    """The per-task call limit is exhausted."""


@dataclass(frozen=True)
class Unit:
    """One thing the decision model judges: a file, a file section, or an image."""

    label: str
    path: str
    kind: str  # "text" or "image"
    payload: str  # the text, or a data URL for an image
    lines: Optional[tuple[int, int]] = None

    @property
    def digest(self) -> str:
        return hashlib.sha256(f"{self.label}\0{self.payload}".encode()).hexdigest()


Skipped = tuple[str, str]  # (label, reason)


# --- Reading files ------------------------------------------------------------


def _ooxml_text(data: bytes) -> str:
    """Paragraph-separated text of one Office Open XML part."""
    pieces: list[str] = []
    for element in ElementTree.fromstring(data).iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "t" and element.text:
            pieces.append(element.text)
        elif tag in ("p", "br"):
            pieces.append("\n")
        elif tag == "tab":
            pieces.append("\t")
    return re.sub(r"\n\s*\n+", "\n", "".join(pieces)).strip()


def _numbered(names: list[str], pattern: str) -> list[str]:
    matching = [name for name in names if re.fullmatch(pattern, name)]
    return sorted(matching, key=lambda name: int(re.findall(r"\d+", name)[-1]))


def _docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        parts = [name for name in ("word/document.xml",) if name in names]
        parts += sorted(
            name
            for name in names
            if re.fullmatch(r"word/(footnotes|endnotes|header\d*|footer\d*)\.xml", name)
        )
        return "\n\n".join(_ooxml_text(archive.read(name)) for name in parts)


def _pptx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        slides = _numbered(archive.namelist(), r"ppt/slides/slide\d+\.xml")
        return "\n\n".join(
            f"[slide {index}]\n{_ooxml_text(archive.read(name))}"
            for index, name in enumerate(slides, 1)
        )


def _xlsx_text(data: bytes) -> str:
    def local(element) -> str:
        return element.tag.rsplit("}", 1)[-1]

    def runs(element) -> str:
        return "".join(t.text or "" for t in element.iter() if local(t) == "t")

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [runs(item) for item in root if local(item) == "si"]
        titles: list[str] = []
        if "xl/workbook.xml" in names:
            root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
            titles = [e.get("name", "") for e in root.iter() if local(e) == "sheet"]
        sheets = []
        for index, name in enumerate(_numbered(names, r"xl/worksheets/sheet\d+\.xml")):
            rows = []
            for row in ElementTree.fromstring(archive.read(name)).iter():
                if local(row) != "row":
                    continue
                cells = []
                for cell in row:
                    if local(cell) != "c":
                        continue
                    value = next((c for c in cell if local(c) == "v"), None)
                    inline = next((c for c in cell if local(c) == "is"), None)
                    text = value.text or "" if value is not None else ""
                    if (
                        cell.get("t") == "s"
                        and text.isdigit()
                        and int(text) < len(shared)
                    ):
                        text = shared[int(text)]
                    elif inline is not None:
                        text = runs(inline)
                    cells.append(text)
                if any(cells):
                    rows.append("\t".join(cells))
            title = titles[index] if index < len(titles) else name
            sheets.append(f"[sheet {title}]\n" + "\n".join(rows))
        return "\n\n".join(sheets)


def _pdf_text(data: bytes) -> Optional[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join(
        f"[page {index}]\n{page.extract_text() or ''}"
        for index, page in enumerate(reader.pages, 1)
    )


def _html_text(text: str) -> str:
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h\d)>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()


def _decode(data: bytes) -> Optional[str]:
    if b"\0" in data[:8192]:
        return None
    text = data.decode("utf-8", errors="replace")
    return text if text.count("�") <= max(8, len(text) // 100) else None


def _image_url(data: bytes, suffix: str) -> Optional[str]:
    mime = IMAGE_TYPES.get(suffix)
    if mime is None:
        from PIL import Image

        buffer = io.BytesIO()
        Image.open(io.BytesIO(data)).convert("RGB").save(buffer, format="PNG")
        data, mime = buffer.getvalue(), "image/png"
    if len(data) > MAX_IMAGE_BYTES:
        return None
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _sections(label: str, path: str, text: str) -> list[Unit]:
    """One unit for a short text; line-ranged sections for a long one."""
    if len(text) <= SECTION_CHARS:
        return [Unit(label, path, "text", text)]
    units: list[Unit] = []
    buffer: list[str] = []
    first = 1

    def flush(last: int) -> None:
        if buffer:
            units.append(
                Unit(
                    f"{label} (lines {first}-{last})",
                    path,
                    "text",
                    "".join(buffer),
                    (first, last),
                )
            )
            buffer.clear()

    lines = text.splitlines(keepends=True)
    size = 0
    for number, line in enumerate(lines, 1):
        if size + len(line) > SECTION_CHARS:
            flush(number - 1)
            first, size = number, 0
        # A single line longer than a section is cut into section-sized parts.
        while len(line) > SECTION_CHARS:
            buffer.append(line[:SECTION_CHARS])
            flush(number)
            first, line = number, line[SECTION_CHARS:]
        buffer.append(line)
        size += len(line)
    flush(len(lines))
    return units


def _email_units(
    label: str, path: str, data: bytes, depth: int
) -> tuple[list[Unit], list[Skipped]]:
    message = email.message_from_bytes(data, policy=email.policy.default)
    header = "\n".join(
        f"{name}: {message[name]}"
        for name in ("From", "To", "Cc", "Date", "Subject")
        if message[name]
    )
    bodies: list[str] = []
    html_bodies: list[str] = []
    units: list[Unit] = []
    skipped: list[Skipped] = []
    for part in message.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        if filename or part.get_content_disposition() == "attachment":
            found, missed = read_units(
                f"{label} > {filename or 'attachment'}",
                path,
                part.get_payload(decode=True) or b"",
                depth + 1,
            )
            units += found
            skipped += missed
            continue
        if part.get_content_maintype() != "text":
            continue
        payload = part.get_payload(decode=True) or b""
        text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        if part.get_content_subtype() == "html":
            html_bodies.append(_html_text(text))
        else:
            bodies.append(text)
    text = "\n\n".join([header, *(bodies or html_bodies)]).strip()
    return (_sections(label, path, text) if text else []) + units, skipped


def read_units(
    label: str, path: str, data: bytes, depth: int = 0
) -> tuple[list[Unit], list[Skipped]]:
    """What the decision model can judge in one file, and what was skipped."""
    suffix = Path(label.rsplit(" > ", 1)[-1]).suffix.lower()
    if suffix in MEDIA_SUFFIXES:
        return [], [(label, "audio/video")]
    if depth > MAX_NESTING:
        return [], [(label, "nested too deep")]
    if suffix in IMAGE_TYPES or suffix in CONVERTED_IMAGE_SUFFIXES:
        try:
            url = _image_url(data, suffix)
        except Exception:  # noqa: BLE001 - unreadable image
            return [], [(label, "unreadable image")]
        if url is None:
            return [], [(label, "image too large")]
        return [Unit(label, path, "image", url)], []
    if suffix == ".zip":
        units: list[Unit] = []
        skipped: list[Skipped] = []
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.infolist():
                    if member.is_dir() or member.filename.startswith("__MACOSX/"):
                        continue
                    found, missed = read_units(
                        f"{label} > {member.filename}",
                        path,
                        archive.read(member),
                        depth + 1,
                    )
                    units += found
                    skipped += missed
        except zipfile.BadZipFile:
            return [], [(label, "unreadable archive")]
        return units, skipped
    try:
        if suffix == ".eml":
            return _email_units(label, path, data, depth)
        if suffix == ".docx":
            text = _docx_text(data)
        elif suffix == ".pptx":
            text = _pptx_text(data)
        elif suffix == ".xlsx":
            text = _xlsx_text(data)
        elif suffix == ".pdf":
            text = _pdf_text(data)
            if text is None:
                return [], [(label, "pdf (no reader installed)")]
        else:
            text = _decode(data)
            if text is None:
                return [], [(label, "binary")]
            if suffix in HTML_SUFFIXES:
                text = _html_text(text)
    except Exception:  # noqa: BLE001 - a corrupt document
        return [], [(label, "unreadable document")]
    if not text.strip():
        return [], [(label, "no text")]
    return _sections(label, path, text), []


def material_files(kit_path: Path) -> list[str]:
    """Kit-relative paths of every task-material file."""
    kit_path = Path(kit_path)
    files = []
    for path in kit_path.rglob("*"):
        relative = path.relative_to(kit_path)
        if relative.parts[0] in EXCLUDED_ENTRIES or any(
            part.startswith(".") for part in relative.parts
        ):
            continue
        if path.is_file() and not path.is_symlink():
            files.append(relative.as_posix())
    return sorted(files)


# --- The Decisions API ----------------------------------------------------------

Decide = Callable[[Unit, str], tuple[float, int]]


def decisions_client(api_key: Optional[str] = None, timeout: float = 60.0) -> Decide:
    """``decide(unit, question) -> (probability, input_tokens)`` over HTTPS."""
    import httpx

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise CorpusSearchError("search_corpus needs OPENAI_API_KEY on the host")
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com"
    base_url = base_url.rstrip("/").removesuffix("/v1")
    client = httpx.Client(
        timeout=timeout,
        headers={"Authorization": f"Bearer {api_key}"},
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=64),
    )

    def decide(unit: Unit, question: str) -> tuple[float, int]:
        if unit.kind == "image":
            content = [
                {"type": "input_text", "text": f"File: {unit.label}"},
                {"type": "input_image", "image_url": unit.payload},
            ]
        else:
            content = [
                {"type": "input_text", "text": f"File: {unit.label}\n\n{unit.payload}"}
            ]
        body = {
            "model": DECISIONS_MODEL,
            "input": [{"role": "user", "content": content}],
            "questions": [
                {"type": "predicate", "name": "answer", "instructions": question}
            ],
        }
        for attempt in range(6):
            try:
                response = client.post(base_url + DECISIONS_PATH, json=body)
            except httpx.HTTPError:
                if attempt == 5:
                    raise
                time.sleep(2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 5:
                    response.raise_for_status()
                retry_after = response.headers.get("retry-after", "")
                time.sleep(float(retry_after) if retry_after.isdigit() else 2**attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            answer = next(a for a in payload["answers"] if a.get("name") == "answer")
            tokens = int((payload.get("usage") or {}).get("input_tokens", 0))
            return float(answer["probability"]), tokens
        raise CorpusSearchError("Decisions API retries exhausted")

    return decide


# --- The tool -----------------------------------------------------------------


class CorpusSearch:
    """Host-side state of the ``search_corpus`` tool for one construction run."""

    def __init__(
        self,
        kit_path: Path,
        *,
        decide: Optional[Decide] = None,
        max_calls: int = MAX_CALLS,
        concurrency: int = 48,
    ):
        self.kit_path = Path(kit_path).resolve()
        # The task materials as shipped, fixed before the Developer starts, so
        # its own notes and outputs never become part of the corpus. Contents
        # are read on the first search.
        self.files = material_files(self.kit_path)
        self._decide = decide
        self.max_calls = max_calls
        self.concurrency = concurrency
        self.calls_used = 0
        self.decisions = 0
        self.input_tokens = 0
        self._units: Optional[list[Unit]] = None
        self._skipped: list[Skipped] = []
        self._cache: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def usage(self) -> dict:
        """Calls, decisions and spend so far, for run metadata."""
        return {
            "model": DECISIONS_MODEL,
            "calls_used": self.calls_used,
            "decisions": self.decisions,
            "input_tokens": self.input_tokens,
            "cost_usd": round(self.input_tokens * USD_PER_INPUT_TOKEN, 4),
        }

    def _load(self) -> list[Unit]:
        if self._units is None:
            units: list[Unit] = []
            for relative in self.files:
                path = self.kit_path / relative
                if Path(relative).suffix.lower() in MEDIA_SUFFIXES:
                    self._skipped.append((relative, "audio/video"))
                    continue
                if not path.is_file():
                    continue
                if path.stat().st_size > MAX_FILE_BYTES:
                    self._skipped.append((relative, "too large"))
                    continue
                found, missed = read_units(relative, relative, path.read_bytes())
                units += found
                self._skipped += missed
            self._units = units
        return self._units

    def search(
        self,
        question: str,
        min_probability: Optional[float] = 0.5,
        path: Optional[str] = None,
    ) -> str:
        if not isinstance(question, str) or not question.strip():
            raise CorpusSearchError("search_corpus requires a question")
        if self.calls_used >= self.max_calls:
            raise CorpusSearchQuotaError(
                f"search_corpus limit of {self.max_calls} calls reached"
            )
        min_probability = 0.5 if min_probability is None else float(min_probability)
        min_probability = min(max(min_probability, 0.0), 1.0)
        prefix = (path or "").strip()
        if prefix == "/workspace" or prefix.startswith("/workspace/"):
            prefix = prefix[len("/workspace") :]
        prefix = prefix.strip("/").removeprefix("./").rstrip("/")

        def under(label: str) -> bool:
            return not prefix or label == prefix or label.startswith(prefix + "/")

        units = [unit for unit in self._load() if under(unit.path)]
        skipped = [(label, reason) for label, reason in self._skipped if under(label)]
        if not units:
            raise CorpusSearchError(f"No readable task-material files under {path!r}")
        self.calls_used += 1
        if self._decide is None:
            self._decide = decisions_client()
        decide = self._decide

        def judge(unit: Unit) -> tuple[Unit, Optional[float], Optional[str]]:
            key = (unit.digest, question)
            with self._lock:
                cached = self._cache.get(key)
            if cached is not None:
                return unit, cached, None
            try:
                probability, tokens = decide(unit, question)
            except Exception as exc:  # noqa: BLE001 - reported per item
                return unit, None, f"{type(exc).__name__}: {exc}"[:200]
            with self._lock:
                self._cache[key] = probability
                self.decisions += 1
                self.input_tokens += tokens
            return unit, probability, None

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            results = list(pool.map(judge, units))
        elapsed = time.monotonic() - started

        judged = sorted(
            ((unit, p) for unit, p, _ in results if p is not None),
            key=lambda item: (-item[1], item[0].label),
        )
        errors = [(unit.label, error) for unit, _, error in results if error]
        hits = [(unit, p) for unit, p in judged if p >= min_probability]

        out_dir = self.kit_path / "corpus_search"
        out_dir.mkdir(exist_ok=True)
        out_path = (
            out_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        )
        out_path.write_text(
            json.dumps(
                {
                    "question": question,
                    "path": path,
                    "min_probability": min_probability,
                    "results": [
                        {
                            "file": unit.path,
                            "item": unit.label,
                            "kind": unit.kind,
                            "lines": unit.lines,
                            "probability": p,
                        }
                        for unit, p in judged
                    ],
                    "not_judged": [
                        {"item": label, "error": error} for label, error in errors
                    ],
                    "skipped": [
                        {"item": label, "reason": reason} for label, reason in skipped
                    ],
                },
                indent=1,
            )
        )

        images = sum(1 for unit in units if unit.kind == "image")
        report = [
            f"Asked {len(units)} items from {len({u.path for u in units})} files "
            f"({len(units) - images} text, {images} images) in {elapsed:.0f}s."
        ]
        if skipped:
            reasons: dict[str, int] = {}
            for _, reason in skipped:
                reasons[reason] = reasons.get(reason, 0) + 1
            report.append(
                "Skipped (not read): "
                + ", ".join(f"{n} {reason}" for reason, n in sorted(reasons.items()))
                + "."
            )
        if errors:
            report.append(f"Could not be judged (API errors): {len(errors)}.")
        report.append(f"Items with probability >= {min_probability:.2f}: {len(hits)}")
        report += [f"{p:.2f}  {unit.label}" for unit, p in hits[:MAX_LISTED]]
        if len(hits) > MAX_LISTED:
            report.append(f"... {len(hits) - MAX_LISTED} more in the saved result.")
        # The call cap is a safety stop, not advertised to the Developer.
        report.append(f"Result for every item: corpus_search/{out_path.name}.")
        return "\n".join(report)
