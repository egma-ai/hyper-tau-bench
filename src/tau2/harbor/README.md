# Running τ^τ-bench on Harbor

`tau2 harbor export` writes the Hyper-τ construction tasks as a
[Harbor](https://github.com/harbor-framework/harbor) dataset (task format
`schema_version = "1.4"`). Each exported task is a plain Harbor task
directory, so `harbor run -p <dataset> -a <agent> -m <model>` benchmarks any
Harbor agent as the Developer, and Harbor's job tooling (attempts,
concurrency, retries, result viewers) applies unchanged. The task's verifier
scores the agent the Developer built on the hidden test conversations, as
`tau2 hyper-tau` does, so one Harbor task covers both loops.

```bash
make harbor-images
uv run tau2 harbor export --output-dir harbor/hyper-tau          # every release task
uv run tau2 harbor export --output-dir harbor/hyper-tau --source-domain retail_plus
harbor run -p harbor/hyper-tau -a codex -m openai/gpt-5.5 --env-file .env
```

## Images

`make harbor-images` builds the two images exported tasks use:

- `tau2-construction-runtime:contract-v7` (`make hyper-construction-image`)
  is the construction runtime of `tau2 hyper-tau`: the agent's container,
  and the sealed `candidate` service that runs the Developer's code.
- `tau2-harbor-runtime:local` (`make harbor-runtime-image`,
  [`docker/harbor-runtime/`](../../../docker/harbor-runtime/)) is the trusted
  runtime: the full source and data, the simulators and the scorers. Agents
  and candidate code never run in it.

Exports name the images they use (`--runtime-image`, `--construction-image`),
so a registry tag works as well as a local one.

## What a task runs

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

Harbor launches the agent with its own flags, and the agent's container adds
the settings the native harnesses run Codex and Claude Code with: Codex reads
them from a system `/etc/codex/config.toml`, which it merges under the config
Harbor writes, and Claude Code from its managed settings
(`/etc/claude-code/managed-settings.json`) and environment. Web search is off
in both, as are sub-agents, plugins, connectors and memory, and Claude Code
gets only the native tools (Harbor's default launch would also give it web
search and fetch, sub-agents, scheduling and worktrees). For Codex, a run's
own agent options (such as `--ak web_search=live`) still take precedence.

Tool calls block until they finish, as in native runs. Local tests, live
experiments and sample scenarios can run for minutes, so the same settings
give both harnesses the native per-call limit, the construction budget
(`MCP_TOOL_TIMEOUT` for Claude Code). Without it, the pinned Codex stops
waiting for a call after five minutes. Agents without MCP support can use the
same tools from a shell with `hyper-tau-tool`.

Harbor's `oracle` agent runs `solution/solve.sh`. Construction has no
reference build, so it submits the starting kit unchanged: a check of the
whole pipeline whose expected reward is about zero.

The agent phase defaults to the task's eight-hour budget
(`--agent-timeout-sec`). Scoring runs up to 32 conversations at once; set
`TAU2_HYPER_INNER_MAX_WORKERS` in the environment Harbor runs in to change
that.

## What differs from `tau2 hyper-tau`

- **The Developer harness is Harbor's agent.** Harbor launches it with its
  own flags and records its trajectory. Codex and Claude Code are already in
  the construction image at their pinned versions, so Harbor uses those
  unless a run asks for another version, with the native settings above; any
  other agent Harbor installs itself. Native runs start Claude Code with
  `--bare`, which in the pinned version leaves it only Bash, Edit and Read of
  its own tools; under Harbor it also has Write and NotebookEdit. The session
  record in the artifact lists the Developer tool calls only.
- **The Developer's own model calls go straight to the provider** through
  Harbor's egress allowlist (extend it with `--allowed-host` at export or
  `--allow-agent-host` at run time) instead of through the model gateway,
  which pins them to the run's model. The models the Developer's agent may
  call are enforced on the trusted side, as in native runs.
- **The Client's opening is fixed per export.** It is part of the
  instruction, so `tau2 harbor export` asks the Client for it once (one call
  to the Client's model, which needs its key) and every trial of the task
  starts from it. Re-exporting reuses it; `--regenerate-client-openings`
  asks again.
- **Candidate sessions share one locked-down service** instead of starting a
  `docker run` container each, because the trusted side has no Docker
  daemon. Each session still gets its own uid, home, and temp directory, but
  sessions share the service's CPU and memory instead of each getting the
  task's container limits.

An agent whose MCP client caps a tool call below the tools' running time can
still run the tasks: export with `--mcp-call-timeout N`, and a call still
running after N seconds returns a job id that the extra `wait_for_result`
tool keeps waiting on. Codex and Claude Code do not need it.

## Keys

Pass the provider keys to Harbor's process (`--env-file .env`). Harbor
forwards them to the trusted sidecar and the verifier only; the agent gets
the keys its own harness needs from Harbor, as for any task.

`banking_knowledge` tasks run their conversations in the domain's default
retrieval configuration (`alltools`), which searches the knowledge base with
OpenAI embeddings, so they also need `OPENAI_API_KEY`. Each container starts
with an empty embeddings cache, so `hyper-runtime` and the verifier each
embed the knowledge base once, before any conversation runs.

## Codex on a ChatGPT plan

Codex can run on a ChatGPT plan instead of an API key. Sign in once on the
machine that runs Harbor, then tell Harbor's Codex agent to use that sign-in:

```bash
codex login
CODEX_FORCE_AUTH_JSON=1 harbor run -p harbor/hyper-tau -a codex -m openai/gpt-5.5 --env-file .env
```

`CODEX_FORCE_AUTH_JSON=1` uses `~/.codex/auth.json`; `CODEX_AUTH_JSON_PATH`
names another file. Harbor copies the file into the agent's container for the
run, and the tasks' allowlist includes the two hosts Codex then calls,
`chatgpt.com` and `auth.openai.com`.

- The plan pays only for the Developer's own model calls. The Client, the
  user simulator and the agent the Developer builds still use the API keys in
  `.env`.
- The plan's usage limits apply: Codex stops when it reaches one, which ends
  the agent phase early.
- Sign-ins renew with single-use tokens. If Codex renews the copy in a trial's
  container, the copy on your machine can no longer renew itself, and your
  own Codex will ask you to sign in again; trials running at once on one
  sign-in can run into the same problem with each other.
