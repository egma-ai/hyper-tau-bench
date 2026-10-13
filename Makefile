# Default target
.PHONY: all
all: help

## Clean up generated files and virtual environment
.PHONY: clean
clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf build

## Run core tests (requires: uv sync --extra dev)
.PHONY: test
test:
	uv run pytest tests/ --ignore=tests/test_voice --ignore=tests/test_streaming --ignore=tests/test_gym --ignore=tests/test_domains/test_banking_knowledge

## Run voice and streaming tests (requires: uv sync --extra dev --extra voice)
.PHONY: test-voice
test-voice:
	uv run pytest tests/test_voice tests/test_streaming -m "not full_duplex_integration"

## Run knowledge/banking tests (requires: uv sync --extra dev --extra knowledge)
.PHONY: test-knowledge
test-knowledge:
	uv run pytest tests/test_domains/test_banking_knowledge

## Run gymnasium tests (requires: uv sync --extra dev --extra gym)
.PHONY: test-gym
test-gym:
	uv run pytest tests/test_gym

## Run all tests (requires: uv sync --all-extras)
.PHONY: test-all
test-all:
	uv run pytest tests/

## Start the Environment CLI for interacting with domain environments
.PHONY: env-cli
env-cli:
	uv run python -m tau2.environment.utils.interface_agent

## Lint code with ruff
.PHONY: lint
lint:
	uv run ruff check .

## Format code with ruff
.PHONY: format
format:
	uv run ruff format .

## Lint and fix issues automatically
.PHONY: lint-fix
lint-fix:
	uv run ruff check --fix .

## Run both linting and formatting
.PHONY: check-all
check-all: lint format

## Build the pinned Hyper-tau native construction image
.PHONY: hyper-construction-image
hyper-construction-image:
	REVISION=$$(git rev-parse HEAD); docker build \
		--build-arg TAU2_SOURCE_REVISION=$$REVISION \
		-f docker/hyper-construction/Dockerfile \
		-t tau2-construction-runtime:contract-v7 \
		-t tau2-construction-runtime:$$REVISION .

## Build the trusted runtime image that exported Harbor tasks run their sidecars and verifiers in
.PHONY: harbor-runtime-image
harbor-runtime-image:
	REVISION=$$(git rev-parse HEAD); docker build \
		--build-arg TAU2_SOURCE_REVISION=$$REVISION \
		-f docker/harbor-runtime/Dockerfile \
		-t tau2-harbor-runtime:local \
		-t tau2-harbor-runtime:$$REVISION .

## Build every image exported Harbor tasks use
.PHONY: harbor-images
harbor-images: hyper-construction-image harbor-runtime-image

## Build the Harbor images and push them to REGISTRY (e.g. ghcr.io/<owner>), tagged with the commit
.PHONY: push-harbor-images
push-harbor-images:
	@test -n "$(REGISTRY)" || { echo "Set REGISTRY, e.g. make push-harbor-images REGISTRY=ghcr.io/<owner>"; exit 1; }
	@git diff --quiet HEAD -- || { echo "Commit your changes first: the pushed images are tagged with the commit"; exit 1; }
	$(MAKE) harbor-images
	REVISION=$$(git rev-parse HEAD); for image in tau2-construction-runtime tau2-harbor-runtime; do \
		docker tag $$image:$$REVISION $(REGISTRY)/$$image:$$REVISION && \
		docker push $(REGISTRY)/$$image:$$REVISION || exit 1; \
	done

## Install pre-commit hooks
.PHONY: setup-hooks
setup-hooks:
	uv run pre-commit install

## Display online help for commonly used targets in this Makefile
.PHONY: help
help:
	@awk '/^[a-zA-Z_\/\.0-9-]+:/ {        \
		nb = sub( /^## /, "", helpMsg );  \
		if (nb)                           \
			print  $$1 "\t" helpMsg;      \
	}                                     \
	{ helpMsg = $$0 }' $(MAKEFILE_LIST) | \
	column -ts $$'\t' |                   \
	expand -t 1 |                         \
	grep --color '^[^ ]*'
