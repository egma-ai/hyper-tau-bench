# Harbor runtime

The trusted image for the Harbor tasks `tau2 harbor export` writes
(see [`src/tau2/harbor/README.md`](../../src/tau2/harbor/README.md)):

```bash
make harbor-runtime-image   # tags tau2-harbor-runtime:local and :<commit>
```

It holds the full tau2 source and data — domains, Hyper-τ tasks and evidence,
and `model_routing.toml` — with the `knowledge` extra and the agentic-shell
sandbox tools (`sandbox-runtime`, ripgrep, bubblewrap, socat) that
`banking_knowledge` environments need, and runs only on the trusted side of a
task: the `hyper-runtime` sidecar and the verifier. Harbor agents and
Developer-written candidate code never run in it; they use the construction
runtime image (`make hyper-construction-image`).

Unlike the construction image, this one must carry `data/tau2/domains`, so
the build uses its own ignore file (`Dockerfile.dockerignore`) instead of the
repository's root `.dockerignore`.

Rebuild it whenever you pull a new version of the repository: an exported task
names the image, and the runtime, the hidden tasks, and the scorer all come
from it.
