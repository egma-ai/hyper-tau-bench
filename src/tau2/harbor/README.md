# Running τ^τ-bench on Harbor

`tau2 harbor` exports the benchmark as [Harbor](https://github.com/harbor-framework/harbor)
datasets (task format `schema_version = "1.4"`). Each exported task is a plain
Harbor task directory, so `harbor run -p <dataset> -a <agent> -m <model>`
runs it with any Harbor agent, and Harbor's job tooling (attempts,
concurrency, retries, result viewers) applies unchanged.

| Command | The Harbor agent plays | One task |
|---------|------------------------|----------|
| `tau2 harbor export-outer` | the Developer | one Hyper-τ construction task |
| `tau2 harbor export-inner` | the customer-service agent | one τ³ conversation |

Both need two images:

```bash
make harbor-images
```

- `tau2-construction-runtime:contract-v7` (`make hyper-construction-image`)
  is the construction runtime of `tau2 hyper-tau`: the agent's container in
  construction tasks, and the sealed `candidate` service that runs the
  Developer's code.
- `tau2-harbor-runtime:local` (`make harbor-runtime-image`,
  [`docker/harbor-runtime/`](../../../docker/harbor-runtime/)) is the trusted
  runtime: the full source and data, the simulators and the scorers. Agents
  and candidate code never run in it.

Exports name the images they use (`--runtime-image`, `--construction-image`),
so a registry tag works as well as a local one.

## Construction tasks (outer loop)

```bash
uv run tau2 harbor export-outer --output-dir harbor/hyper-tau          # every release task
uv run tau2 harbor export-outer --output-dir harbor/hyper-tau --source-domain retail_plus
harbor run -p harbor/hyper-tau -a codex -m openai/gpt-5.5 --env-file .env
```

The instruction is the brief `tau2 hyper-tau` gives its Developer harness,
plus how to reach the tools in this environment. The task runs four
containers:

| Container | Image | Holds | Network |
|-----------|-------|-------|---------|
| `main` | construction runtime | the agent, the kit in `/workspace`, and a stdio MCP stub that forwards the `hyper_tau` tools | Harbor's agent-phase allowlist: model providers only |
| `hyper-runtime` | trusted runtime | the Client simulator, the hidden task partitions, the model keys; serves `run_local_test`, `talk_to_client`, `run_live_experiment`, `run_sample_scenarios`, `submit` | model providers, and the internal sandbox network |
| `candidate` | construction runtime | nothing: it runs one sealed session of the Developer's code per connection (`candidate_host`), read-only workspace and root filesystem, per-session uids, no capabilities beyond those | internal sandbox network only |
| verifier | trusted runtime | the submission snapshot, with a fresh `candidate` service | model providers |

`hyper-runtime` builds the kit into the shared `/workspace` volume before the
agent starts. The agent's tool calls reach it through request and response
files in a second shared volume (the same file protocol the native harnesses
use), so the agent's container never holds hidden state or provider keys.
Candidate code reaches models only through the trusted side, exactly as in
sealed scoring.

Construction ends when the agent calls `submit` or when the agent phase ends
(Harbor then runs the task's `finalize` collect hook in `hyper-runtime`).
Either way the runtime seals the Client API deployment, snapshots
`/workspace`, and records the session. The snapshot and record are the
task's artifact; the verifier unpacks the snapshot into its own workspace and
runs `SandboxOrchestrator.score_submission`, the same code that scores a
`tau2 hyper-tau` run: the hidden test partition through the sealed inner
loop, against the deployment the Developer left in place, with the task's
performance requirements applied.

`reward.json` holds `reward` (the final Hyper-τ reward) and its components
`quality_reward`, `performance_reward` and `performance_penalty`.
`outer_loop_result.json`, next to it in the verifier logs, is the full
`OuterLoopResult` that `tau2 hyper-tau` records, including every scored
conversation.

What differs from `tau2 hyper-tau`:

- **The Developer harness is Harbor's agent.** Harbor installs and configures
  it, so its version and settings are Harbor's, not the ones pinned in the
  construction image, and Harbor records its trajectory. The session record
  in the artifact lists the Developer tool calls only.
- **The Developer's own model calls go straight to the provider** through
  Harbor's egress allowlist (extend it with `--allowed-host` at export or
  `--allow-agent-host` at run time) instead of through the model gateway.
- **The Client's opening is fixed per export.** It is part of the
  instruction, so `export-outer` asks the Client for it once (one call to
  the Client's model, which needs its key) and every trial of the task starts
  from it. Re-exporting reuses it; `--regenerate-client-openings` asks again.
- **Candidate sessions share one locked-down service** instead of starting a
  `docker run` container each, because the trusted side has no Docker
  daemon. Each session still gets its own uid, home, and temp directory.
- **Long tool calls and Codex.** Codex gives an MCP call 60 seconds and
  Harbor's generated Codex config cannot raise that, while local tests, live
  experiments, and sample scenarios can run for minutes. A call still
  running after 50 seconds therefore returns a job id, and the extra
  `wait_for_result` tool keeps waiting for it (`--mcp-call-timeout`; 0
  disables). Agents without MCP support can use the same tools from a shell
  with `hyper-tau-tool`.

Harbor's `oracle` agent runs `solution/solve.sh`. Construction has no
reference build, so it submits the starting kit unchanged: a check of the
whole pipeline whose expected reward is about zero.

The agent phase defaults to the task's eight-hour budget
(`--agent-timeout-sec`). Scoring runs up to 32 conversations at once; set
`TAU2_HYPER_INNER_MAX_WORKERS` in the environment Harbor runs in to change
that.

## Conversation tasks (inner loop)

```bash
uv run tau2 harbor export-inner --domain retail_plus --output-dir harbor/tau3-retail_plus
uv run tau2 harbor export-inner --domain banking_knowledge --output-dir harbor/tau3-banking \
    --retrieval-config bm25
harbor run -p harbor/tau3-retail_plus -a claude-code -m anthropic/claude-sonnet-5 --env-file .env
```

The agent gets the domain policy as its instruction and talks to the
customer only through the `tau2` MCP server: `start_conversation` returns the
customer's first message, `send_message_to_user` returns the customer's
reply, `end_conversation` ends it, and the domain tools act on the
customer's records. The agent's container holds nothing else. The
`tau-runtime` sidecar runs the tau2 orchestrator, the user simulator, and the
domain environment; the verifier scores the recorded conversation with the
tau2 evaluator (`--evaluation-type`, default `all`), and `reward.json` holds
the task reward and its breakdown.

The user simulator defaults to Hyper-τ's inner-loop user model. Set
`TAU2_HARBOR_USER_LLM` and `TAU2_HARBOR_USER_LLM_ARGS` (JSON) in Harbor's
environment to swap it for a run without re-exporting. `oracle` replays the
task's reference actions, so it should score 1.0.

The canonical `airline` and `retail` domains are frozen legacy baselines;
export `airline_plus` and `retail_plus` (or pass `--allow-legacy-domain` for
an ablation).

## Keys

Pass the provider keys to Harbor's process (`--env-file .env`). Harbor
forwards them to the trusted sidecars and the verifier only; the agent gets
the keys its own harness needs from Harbor, as for any task.

`banking_knowledge` construction tasks run their conversations in the
domain's default retrieval configuration (`alltools`), which searches the
knowledge base with OpenAI embeddings, so they also need `OPENAI_API_KEY`.
Each container starts with an empty embeddings cache, so `hyper-runtime` and
the verifier each embed the knowledge base once, before any conversation
runs. Conversation tasks use the retrieval configuration chosen at export
(`bm25` unless `--retrieval-config` says otherwise).
