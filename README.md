# Legoizer

**Build, inspect, and train recommendation models visually.**

Legoizer adds a React Flow editor and a FastAPI service to the real
[Legommenders](https://github.com/Jyonn/Legommenders) framework. Drag components,
connect encoders and predictors, validate the graph, export a real YAML model,
and run training and evaluation on your own machine or GPU server. The interface
supports English and Simplified Chinese.

The repository contains both the visual application and its underlying training
code. Datasets, pretrained weights, environments, and previous training runs are
not included. **You do not need a `.data` file:** Studio downloads and prepares
missing supported datasets when you start training or explicitly acquire resources.

## Requirements

- Linux (Ubuntu is the reference platform), or Windows with WSL2. Run both the
  installation and application inside WSL2. Native Windows and macOS are not
  supported by this release; the resource preparation code uses `fcntl`.
- **Python 3.11**, including `venv` and `pip`. Use this version for the pinned
  NumPy/Pandas stack; newer Python versions are not covered by this release.
- **Node.js 20 or newer and npm** to build/develop the frontend. Use a maintained
  LTS release. Node is not needed to serve an already built frontend.
- Git and `curl`; internet access to Python/npm registries and the selected data
  and model providers. Corporate proxies must also be available to subprocesses.
- A writable checkout with enough disk space for dependencies, temporary files,
  datasets, and the selected model. Large language models can need many GB of
  RAM/VRAM and disk. Start with the CPU demo or a small model.
- For GPU training: an NVIDIA GPU, a driver compatible with your PyTorch CUDA
  wheel, and enough free VRAM. The supplied GPU example uses CUDA 12.8 wheels;
  you do not need a separate CUDA toolkit for the prebuilt PyTorch wheel.

On Ubuntu/WSL2, install Git, curl, and Python 3.11 with its venv support through
an appropriate package source, or use Conda/Miniforge to provide Python 3.11.
Installing those system tools is separate from this project's setup script.

## Run on your own computer (CPU)

```bash
git clone https://github.com/zhhaon/Legoizer.git
cd Legoizer

# Creates .venv, installs Python dependencies, and builds the frontend.
bash studio/setup.sh --torch cpu

# Optional environment report; performs no data/model downloads.
bash studio/doctor.sh

# Serves the frontend and API together, in the foreground.
bash studio/run.sh
```

Open **http://127.0.0.1:8000/**. API documentation is at
**http://127.0.0.1:8000/docs**. If `python3.11` is not on PATH, pass an interpreter:

```bash
bash studio/setup.sh --python /path/to/python3.11 --torch cpu
```

No global Python environment is modified. The installer uses `.venv/` and stores
its temporary files and package caches under `studio/backend/workspace/install/`.
It installs application dependencies only; datasets and model checkpoints are
acquired later. You do not need to run `process.py` before using Studio.

If port 8000 is occupied, run `BACKEND_PORT=8123 bash studio/run.sh` and open
`http://127.0.0.1:8123/`.

## Run on your own NVIDIA GPU server

Clone the repository on the **GPU server**, then run there:

```bash
cd /path/to/Legoizer
nvidia-smi
bash studio/setup.sh --torch cu128
bash studio/doctor.sh --require-cuda
BACKEND_PORT=8000 bash studio/run.sh
```

If your driver requires a different wheel, select a compatible PyTorch 2.9.1
index with `TORCH_INDEX_URL`. Do not install a CPU wheel and expect CUDA to be
available. The application uses one backend worker because process/run state
is managed in memory and persisted to local files.

On **your computer**, open a separate terminal and create an SSH tunnel:

```bash
ssh -N -L 127.0.0.1:8000:127.0.0.1:8000 YOUR_USER@YOUR_GPU_SERVER
```

Keep both terminals open and visit **http://127.0.0.1:8000/** on your computer.
If local port 8000 is busy, use `127.0.0.1:18000:127.0.0.1:8000` and browse
`http://127.0.0.1:18000/`. The browser runs locally; resource preparation,
training, checkpoints, and evaluation all run on the GPU server.

In Studio's **Training** settings, select **CUDA** and the intended GPU index.
The default training device is CPU; installing a CUDA wheel alone does not
change a saved graph's device. Reduce batch size or choose a smaller encoder
if GPU memory is insufficient. When using `CUDA_VISIBLE_DEVICES`, GPU indices
in Studio refer to the visible devices in that process.

## First model and training run

1. Open **NAML Demo** from the welcome screen.
2. Inspect the graph. Select components to change their properties, or replace
   the user encoder with another compatible component.
3. Use **Validate** and **Generate Model** to inspect/export the YAML.
4. Select **CPU** for a small local run, or **CUDA** on your GPU server. Click
   **Train**. The log shows data downloads, tokenization, embedding preparation,
   and then the real trainer's output.
5. Inspect evaluation metrics and compare completed runs. Saved graphs, run
   history, and checkpoints remain in this checkout.

The demo samples up to 650 real MIND-small users with a fixed seed. It still
needs the MIND-small source data and the selected embedding resources. The
initial GloVe archive is approximately **862 MB**, with additional space needed
for extraction and converted embeddings. The first run can take substantially
longer than subsequent runs. Demo results are not substitutes for full-dataset
experimental comparisons.

Use **Cancel run** to stop preparation/training. Cancel active runs before
stopping the backend. Press **Ctrl+C** in the launch terminal, or use another
terminal in the same checkout:

```bash
bash studio/stop.sh
```

The stop command targets the service manager for this checkout; it does not
use machine-wide process-name matching. Logs are in `studio/logs/`.

## Automatic data and optional local configuration

Studio reuses compatible processed tables first. Otherwise it uses valid local
raw data configured in `.data`, or downloads the supported source and preprocesses
it. Missing `.data` is normal and does not produce a configuration warning.
Incomplete MIND/xMIND/RecBench local paths produce a log message and fall back
to acquisition. Upstream failures remain visible errors, not fabricated data.

You only need the optional override if you already have local data, work offline,
or need a particular published split:

```bash
cp .data.example .data
# Edit .data and uncomment only entries for files you actually have.
```

Restart the backend after changing `.data` or `.model`. The standalone upstream
`process.py` CLI still expects its raw-data path configuration; Studio's automatic
preparation is implemented by the Studio backend, not by that CLI.

See [Data sources and offline use](docs/DATA.md) for directory layouts, download
locations, dataset coverage, and split limitations. The tracked `.model` file
contains public Hugging Face model aliases; edit it locally if you need a different
repository or an existing model directory.

Gated/private models require provider approval and your own Hugging Face read
Token. Submit it through localhost, the SSH tunnel above, or configured HTTPS.
Do not put it in YAML, `.model`, README files, or Git. Training credentials are
passed to the child process and excluded from saved graphs and run records.
Downloading a model does not grant access to other restricted resources.

The **Official Benchmark** view retrieves results from an external API. An
unavailable API does not prevent local graph editing and training; benchmark
results are separate from your local runs.

## Configuration and development

The default service is local and single-user. There is no application login,
tenant isolation, or public job quota. Keep loopback access and use SSH for remote
servers. Only bind to another interface when you have arranged appropriate
network access controls; see [Development and deployment](docs/DEVELOPMENT.md).

| Variable | Default | Purpose |
| --- | --- | --- |
| `STUDIO_PYTHON` | `.venv/bin/python`, then active environment | Explicit runtime interpreter |
| `STUDIO_NODE_BIN` | Node on PATH | Directory containing `node` and `npm` |
| `BACKEND_PORT` | `8000` | API and built frontend port |
| `STUDIO_HOST` | `127.0.0.1` | Bind address for backend and development frontend |
| `STUDIO_DEV` | `0` | Set `1` to launch Vite alongside the backend |
| `FRONTEND_PORT` | `5173` | Vite development port |
| `STUDIO_API_URL` | `http://127.0.0.1:<BACKEND_PORT>` | Vite's backend proxy target |
| `NPM_REGISTRY` | `https://registry.npmjs.org` | Registry used by installation scripts |
| `TORCH_INDEX_URL` | Selected CPU/CUDA wheel index | Override during setup only |

The scripts read environment variables; they do not automatically load a `.env`
file. Export settings in your shell or pass them before a command.

For tests and live frontend editing:

```bash
bash studio/setup.sh --torch cpu --dev
.venv/bin/python -m pytest studio/backend/tests -q
(cd studio/frontend && npm test && npm run build)
STUDIO_DEV=1 bash studio/run.sh
```

Development UI: `http://127.0.0.1:5173/`. Vite proxies `/api` to the backend.
Rebuild the frontend after changes when using the default single-port mode.
See the [Studio guide](studio/README.md) for editor architecture, API, component
extension, and optional browser tests. CI runs CPU backend tests and frontend
checks; it does not download datasets or run GPU training. See the
[validation notes](docs/VALIDATION.md) for the checks actually completed for
this prepared release and the remaining platform coverage.

## Troubleshooting

- **Missing Python modules or import errors:** use `.venv/bin/python`, rerun setup,
  then `bash studio/doctor.sh`. Do not combine an old global Transformers/PyTorch
  stack with this release's pinned runtime.
- **CUDA unavailable:** inspect `nvidia-smi`, the selected interpreter, and its
  PyTorch wheel. Run `bash studio/doctor.sh --require-cuda` on the GPU server.
- **Downloads fail:** inspect the training log, provider access, proxy settings,
  and available disk space. Use `.data` to supply existing data if necessary.
  Large installation temporary files live under the project rather than `/tmp`.
- **Page says frontend is not built:** run `npm ci && npm run build` in
  `studio/frontend`, then restart the backend.
- **Port is in use:** select a different `BACKEND_PORT` and update the SSH tunnel.
- **Token rejected over plain remote HTTP:** use localhost through SSH, or HTTPS.
- **You changed a local data/model path:** restart the backend to refresh cached
  alias configuration. Dataset-indexed embedding matrices must match their
  dataset and cannot always be downloaded automatically.

## Repository, attribution, and license

```text
studio/frontend/        React + TypeScript visual editor
studio/backend/         FastAPI, registry, validator, compiler, executor, tests
studio/scripts/         Runtime discovery, checks, service management
config/                 Model, data, embedding, and experiment YAML
model/, loader/         Legommenders model/runtime components
trainer.py, tester.py   Real training and evaluation entry points
docs/                   Data, development, and release guides
```

This project is derived from [Jyonn/Legommenders](https://github.com/Jyonn/Legommenders).
It retains the original [MIT license and copyright notice](LICENSE). The archived
[upstream README](README.legommenders.md) documents its models, CLI, publications,
and citations. Legoizer adds visual programming, resource acquisition, run
management, bilingual UI, and compatibility/training fixes; it is not the upstream
project's official hosted service.

Dataset and model assets are obtained from their respective publishers and are
not bundled in this source repository. See their access and usage terms before
using or redistributing those assets. See [Release preparation](docs/RELEASING.md)
before publishing your own copy.
