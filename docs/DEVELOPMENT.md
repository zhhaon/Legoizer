# Development and deployment

## Runtime and dependencies

Python 3.11 is the reference interpreter. `requirements.txt` declares the model
runtime; `studio/backend/requirements.txt` declares the API runtime;
`requirements-dev.txt` adds backend test tooling. `constraints-python311.txt`
pins the direct runtime dependencies and selected compatibility dependencies
from the verified stack. This is a constraints file, not a hash-locked complete
transitive dependency snapshot. Review updates together with model-layer tests.

`setup.sh` installs PyTorch first from the selected official CPU or CUDA wheel
index, then the remaining dependencies with those constraints. It creates or
uses `.venv/`; an existing Python 3.11 Conda environment at that prefix also works.
To supply Python without a system `python3.11`, for example:

```bash
conda create --prefix "$PWD/.venv" --override-channels -c conda-forge python=3.11 pip
bash studio/setup.sh --torch cpu --dev
```

The scripts never accept Conda channel terms on your behalf. Choose channels
appropriate for your own environment. Set `STUDIO_BOOTSTRAP_PYTHON` or use
`--python` when creating a normal venv. `STUDIO_PYTHON` selects the interpreter
for run/doctor commands, not the interpreter installed by setup.

## Frontend development

```bash
STUDIO_DEV=1 bash studio/run.sh
```

The backend listens on port 8000 and Vite on 5173 by default. The frontend uses
relative `/api` requests; Vite proxies them to `STUDIO_API_URL`. In the normal
built mode, FastAPI serves `studio/frontend/dist` and the API from one origin.
When either service in the combined launcher fails, the manager stops its other
child. Each checkout has its own lock and manager identity under workspace.

Standalone foreground processes are also available:

```bash
bash studio/run_backend.sh
# In a different terminal:
bash studio/run_frontend.sh
```

Use Ctrl+C in each standalone terminal; `stop.sh` targets only the combined
`run.sh` manager. Stop active training before shutting down the backend.
Do not run several backend workers against one workspace.

## Backend tests and browser acceptance

```bash
.venv/bin/python -m pytest studio/backend/tests -q
(cd studio/frontend && npm test && npm run build)
```

The regular tests do not intentionally download datasets or model checkpoints.
Run them with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to enforce that boundary.
Some tests write generated configurations and graph/run fixtures under the
ignored workspace; they do not modify shipped YAML files.

Optional browser scripts in `studio/tests/` use Playwright. Install Playwright
separately outside the release source tree or as a temporary local dependency,
and set `PLAYWRIGHT_CORE` to its module directory and `CHROME_PATH` if using your
own browser. Set `STUDIO_URL` to the running app. Some scripts start real training
and download resources; read the script before running it. The i18n browser test
only checks the interface and compilation. Browser tests are not CI prerequisites.

## Remote access and HTTPS

The default loopback listener is intended for a local user or SSH forwarding.
There is no built-in account system. CORS is not an authentication mechanism.
Do not treat this backend as a multi-user public training service.

For an access-controlled LAN deployment, explicitly set `STUDIO_HOST`:

```bash
STUDIO_HOST=0.0.0.0 BACKEND_PORT=8000 bash studio/run.sh
```

Arrange your network access controls first. For a browser deployed on a separate
origin, set `STUDIO_CORS_ORIGINS` to a comma-separated list of exact allowed
origins. Same-origin builds and the Vite proxy need no CORS allowance.

Native HTTPS can be enabled with PEM files matching the name/IP visitors open:

```bash
export STUDIO_HOST=127.0.0.1
export STUDIO_HTTPS_PORT=8443
export STUDIO_TLS_CERT=/path/to/certificate.pem
export STUDIO_TLS_KEY=/path/to/private-key.pem
bash studio/run.sh
```

The single Hypercorn worker serves HTTPS and the normal HTTP listener with the
same run state. Bind externally only for your intended deployment. Configure
trusted proxy headers carefully if terminating TLS elsewhere; do not assume a
forwarded `https` header is trustworthy. The localhost SSH tunnel documented in
the root README is the simplest remote option and supports Hugging Face tokens.
Never commit certificates containing private keys or tokens.

## Component extension

Keep the existing `model/`, `loader/`, and `config/` layout: the backend discovers
real classes and resolves shipped YAML rather than maintaining a separate model
implementation. The registry overlays semantic ports and parameter descriptions.
See `studio/README.md` for the compiler, validator, registry, and API contracts.
The trainer compatibility changes in this fork are required by this runtime.
