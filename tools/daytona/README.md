# Running Hyper-τ on Daytona

`launch.py` runs release tasks in parallel, one [Daytona](https://www.daytona.io)
sandbox per task. Each sandbox is a Docker-in-Docker VM (`docker:28.3.3-dind`,
the base Harbor uses on Daytona) that runs the unmodified sealed runner
(`tau2 hyper-tau`) for one task, so a task is isolated exactly as it is on a
local Docker host: the Developer's container has no internet, and the model
gateway sidecar is its only exit.

`runner.sh` is what runs inside each sandbox: it starts `dockerd`, installs the
host (`uv`, plus a compiler for `psutil`, which has no musl wheel), applies
this checkout's diff on top of the upstream base commit, builds the
construction image and runs the task.

## Requirements

- A Daytona org with room for 4 vCPU / 8 GiB / 10 GiB per concurrent task
  (Tier 3 runs all 53 release tasks at once).
- `DAYTONA_API_KEY`, `OPENAI_API_KEY` and `OPENROUTER_API_KEY` in the
  environment or `.env`. The launcher writes the provider keys into each
  sandbox's `.env`; nothing is stored in sandbox metadata.
- Your changes committed or at least tracked: the launcher ships
  `git diff --binary <base>`, where `<base>` defaults to
  `git merge-base HEAD origin/main` and must exist upstream.

## Usage

```bash
uv run --with "daytona>=0.210.0" python tools/daytona/launch.py \
    --run-name gpt61sol-max --tasks all --concurrency 10 \
    --developer-llm gpt-6.1-sol --developer-effort max
```

- `--tasks` takes `all` or a comma list of task ids or 3-digit release slots
  (`001,016,021`). `--dry-run` shows what would run.
- Results go to `data/simulations/hyper_tau_daytona/<run-name>/`: `state.json`,
  one directory per task (logs and the run recording), `summary.json` and
  `summary.md`. Failed builds count as 0 in `overall_pct`, as in a submission.
- Re-running the same command resumes: finished tasks are skipped and
  sandboxes that are still running are re-attached. `--summarize-only`
  rebuilds the summary from `state.json`.
- Every sandbox has a 12-hour wall-clock TTL, so an interrupted launcher can
  never leave sandboxes running indefinitely.

## Billing the Developer to a ChatGPT plan

`--developer-auth chatgpt` (the default) serves the Developer model through
Codex's ChatGPT-plan backend instead of an API key; the simulators, judges
and the agents being scored still use your API keys.

Sign in once. `chatgpt_login.py` keeps the Codex login in a small Daytona
sandbox, the login keeper (1 vCPU, stopped when idle, never auto-deleted),
and every launch, on any machine with the Daytona key, fetches its access
token from there:

```bash
uv run --with "daytona>=0.210.0" python tools/daytona/chatgpt_login.py login
```

This prints a device code to approve at https://auth.openai.com/codex/device
once. Afterwards:

- The keeper is the only place that ever refreshes the login (with Codex's
  own refresh flow, under a lock). Refresh tokens are single-use, so a login
  refreshed in two places logs both out; launchers never refresh it.
- Each new sandbox gets a token valid for at least its 12-hour lifetime. The
  keeper refreshes the login once it has less than 48 hours left, so runs of
  any length and later runs need no new sign-in.
- `chatgpt_login.py status --usage` shows the token expiry and the plan's
  usage windows without spending a model turn. `store --auth <auth.json>`
  moves an existing dedicated Codex login into the keeper instead of signing
  in (its local copy becomes access-only). `forget --yes` deletes the keeper.
- If OpenAI ever revokes the login (a password change, or signing out of all
  sessions), run `chatgpt_login.py login --replace`.

`--chatgpt-auth <auth.json>` uses a local Codex login instead. The launcher
never refreshes it and refuses new sandboxes once it has less than 13 hours
left.

Either way, only the access token and account id reach a task sandbox
(never the refresh token). There, the token is mounted read-only into the
model-gateway sidecar and never enters the Developer's container.

Caveats: OpenAI recommends API keys for automated Codex use, so batch runs on
a personal plan are subject to its fair-use limits; every sandbox draws on
the same plan allowance; and scores produced this way should be labelled as
served through the ChatGPT backend.
