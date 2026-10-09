#!/bin/sh
# Runs ONE Hyper-tau task inside a Daytona Docker-in-Docker sandbox
# (docker:28.3.3-dind, Alpine). Uploaded and started by tools/daytona/launch.py,
# which polls $H/state and collects $H/*.log and the run recording afterwards.
#
# Inputs (all under $H, written by the launcher before this starts):
#   job.env           TASK_ID, BASE_COMMIT, REPO_URL, DEV_LLM, DEV_EFFORT, DEV_AUTH
#   branch.patch      `git diff --binary BASE_COMMIT` of the launcher's checkout
#   dotenv            the repo .env (provider keys, TAU2_CHATGPT_AUTH_FILE)
#   chatgpt-auth.json access token + account id only (when DEV_AUTH=chatgpt)
set -u
H=/root/hyper
REPO=/root/htb
mkdir -p "$H"
. "$H/job.env"

log() { echo "[$(date -u +%H:%M:%S)] $*" >> "$H/runner.log"; }
state() { echo "$1" > "$H/state"; log "state=$1"; }
fail() { state "failed:$1"; exit 1; }

state starting-docker
# Daytona does not run the image entrypoint; start the daemon ourselves.
dockerd-entrypoint.sh dockerd > /var/log/dockerd.log 2>&1 &
for _ in $(seq 1 90); do docker info > /dev/null 2>&1 && break; sleep 1; done
docker info > /dev/null 2>&1 || fail dockerd

state installing
# psutil has no musl wheel, so the host venv needs a compiler and headers.
apk add --no-cache bash git curl python3 python3-dev build-base linux-headers \
    >> "$H/runner.log" 2>&1 || fail apk
curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh \
    >> "$H/runner.log" 2>&1 || fail uv-install

state fetching-code
{
    git init -q "$REPO" \
        && git -C "$REPO" remote add origin "$REPO_URL" \
        && git -C "$REPO" fetch -q --depth 1 origin "$BASE_COMMIT" \
        && git -C "$REPO" checkout -q FETCH_HEAD \
        && { [ ! -s "$H/branch.patch" ] \
            || git -C "$REPO" apply --whitespace=nowarn "$H/branch.patch"; }
} >> "$H/runner.log" 2>&1 || fail code
cp "$H/dotenv" "$REPO/.env" && chmod 600 "$REPO/.env"

state uv-sync
cd "$REPO" || fail code
uv sync --frozen --python python3 >> "$H/runner.log" 2>&1 || fail uv-sync

state building-image
docker build -f docker/hyper-construction/Dockerfile \
    -t tau2-construction-runtime:contract-v7 \
    --build-arg TAU2_SOURCE_REVISION="$BASE_COMMIT+patch" . \
    > "$H/build.log" 2>&1 || fail image-build

state running
uv run --frozen tau2 hyper-tau "$TASK_ID" \
    --developer-harness codex \
    --developer-llm "$DEV_LLM" \
    --developer-reasoning-effort "$DEV_EFFORT" \
    --developer-auth "$DEV_AUTH" \
    --no-display > "$H/run.log" 2>&1
echo $? > "$H/exit_code"
state done
