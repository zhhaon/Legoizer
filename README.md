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
missing supported datasets when you start training or explicitly acquire resources
in server mode. Pages users export a package and prepare resources on their own
execution machine.

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

### If Node.js or Python 3.11 is missing

Run these checks in the same terminal that will run setup:

```bash
node -v
npm -v
python3.11 --version
```

`Missing node` means Node is absent from PATH, not a CUDA error. If Node and npm
are already installed elsewhere, point setup at their **bin directory**:

```bash
STUDIO_NODE_BIN=/path/to/node/bin \
bash studio/setup.sh --python /path/to/python3.11 --torch cu128
```

Both `node -v` and `npm -v` must work from that directory. A copied/broken npm
launcher is not sufficient. Do not split the filename `studio/setup.sh` across
lines; a shell continuation backslash belongs at the end of the preceding line.

For a Conda-based installation, one option is a dedicated bootstrap environment:

```bash
conda create -n legoizer-setup -c conda-forge python=3.11 nodejs=22 -y
conda activate legoizer-setup
python --version
node -v
npm -v

# Run from the cloned Legoizer project root.
bash studio/setup.sh --python "$(command -v python)" --torch cu128
```

`python3.11: command not found` means the default bootstrap interpreter is not
on PATH. Activate a Python 3.11 environment and pass its executable with
`--python "$(command -v python)"`, as in the example above.

Use `--torch cpu` for CPU installation. The selected bootstrap Python creates the
project's `.venv`; activate `.venv` for manual training commands afterwards.
`STUDIO_PYTHON` selects the runtime interpreter, whereas setup's `--python` selects
the interpreter used to create `.venv`. Setup does not replace an existing `.venv`
with a different Python version; it reports a version mismatch instead.

## Choose how to use Legoizer

| User | Where to work | What runs there |
| --- | --- | --- |
| GitHub Pages visitor | Browser, then your own Linux/WSL2 server | Edit/export configs and view static Benchmark in the browser; prepare resources and train on your server |
| Local CPU user | Linux or WSL2 | Run frontend and backend together; prepare resources, train and evaluate through Studio |
| GPU server user | Server plus a local browser over SSH | Run frontend/API, resource preparation and training on the server |
| CLI-only execution user | Linux/WSL2 terminal | Install the project, download or copy a configuration package, prepare resources and train without starting the web service |
| Website maintainer | Node build environment and GitHub Actions | Build/publish the static frontend without a Python web service |

Browser-only visitors do not need Python, Node or CUDA on the computer used to
open Pages. The runtime requirements above apply to the machine that installs,
prepares resources or trains models. CPU/GPU installation and startup steps are
in the following sections. Detailed mode behavior is in
[Studio modes and portable packages](docs/STUDIO_MODES.md).

### GitHub Pages visitors

Use the published site to edit a model and view the static Benchmark. The site
runs in your browser; it cannot inspect your server's GPUs, download resources
onto your server, or start Python training there.

Select a template and dataset, edit the model and training settings, then open
**YAML Preview > Export and run on your server**. Follow the portable workflow
below on your own Linux/WSL2 execution machine. Clicking **Train** in browser-only
mode also opens this guide.

### Users with a frontend and backend

Follow the CPU or NVIDIA GPU installation sections below, then open Studio.
Use **Train** and **Evaluate** to run jobs on the machine hosting the backend;
the browser shows progress and results. The training settings list GPUs detected
on that machine.

You can also open **YAML Preview > Export and run on your server** to download
or copy a package and run it from a terminal, including on a different machine.
The same four-step guide is available in both deployment modes. Its manual GPU
index describes the target execution machine, which may differ from the machine
hosting the backend. Recheck the device before starting a job through Studio.

### Export and run on another machine

Choose one of the two workflows below: use a ZIP with the generated guide, or
copy configuration files and run the manual commands. A Studio web service is
not required on the execution machine for either CLI workflow. The installation
script still requires Python 3.11, Node.js 20 or newer, npm, Git and curl.

The guide preserves the graph's random seed in the generated `--seed` argument.
There is no separate seed input in the training settings or guide. Change the
CLI argument if you need another seed.

#### Download a ZIP and use the generated guide

1. Open **YAML Preview > Export and run on your server** before setting the
   package directory. In the guide, confirm the project path, ZIP path, package
   name and installation option.
2. Replace the prefilled project path `/home/you/projects/Legoizer` with your
   actual absolute server path. The ZIP path defaults to
   `/tmp/legoizer-export.zip`; keep it only if that is where you will upload the
   file. These are editable values used in the generated commands. The browser's
   download directory is separate from the ZIP path on the server.
3. Choose the installation option that matches your target machine:
   - **First installation:** use a new project directory. The generated command
     clones the repository there and runs `studio/setup.sh`. Upload the ZIP
     outside that directory so the clone destination stays available.
   - **Existing environment:** use an installed checkout with an executable
     `.venv/bin/python`. The generated command checks that executable; it does
     not install or repair dependencies. If you installed manually using the
     sections below, select this option.
4. Set the package name, for example `my_model`. Choose **CPU**, or choose
   **CUDA** and enter the GPU index on the target server. Use the index visible
   to that process when `CUDA_VISIBLE_DEVICES` is set. **Auto** exports GPU 0.
   The original browser-only training panel selects the device type; manual
   GPU indices are entered in the guide.
5. Click **Download ZIP** and upload it to the exact server ZIP path you chose.
   Acknowledge completion to advance through the guide. Copy and execute each
   installation/check, resource preparation and training command in a terminal
   on the target machine. **Copy full run flow** copies all three command stages
   together. The webpage does not execute them or detect whether they finished.

The ZIP contains `config/exported/my_model/` with `model.yaml`, `exp.yaml`,
`data.yaml`, `embed.yaml`, `prepare.json`, `graph.json`, `README.txt`,
`RUN_GUIDE.md`, `commands.sh` and `commands.txt`. The package name changes this
path. Compare the catalog revision in `README.txt` when using another checkout.

You can use the step-by-step guide, the copied full flow, or `commands.sh`.
The script stops on errors and executes installation/checks, ZIP extraction,
resource preparation and training. To access it before the first installation,
extract a copy of the ZIP into a temporary directory using Python 3.11, then run
the extracted script with Bash. Keep the original ZIP at the configured server
path: the script extracts it into the project during preparation. For example,
if your uploaded ZIP is `/tmp/legoizer-export.zip` and the package is named
`my_model`:

```bash
python3.11 -m zipfile -e /tmp/legoizer-export.zip /tmp/legoizer-package
bash /tmp/legoizer-package/config/exported/my_model/commands.sh
```

Run the first-installation flow only before that checkout is installed. Once the
project exists, reopen the guide with the existing-environment option and export
updated commands instead of repeating the clone. The generated setup command
uses the default `python3.11` executable; if your interpreter has another path,
use the manual setup instructions with `--python`, then choose the
existing-environment option.

#### Copy configuration files and run manually without a ZIP

Install the project using the CPU or GPU setup instructions below. In the export
guide, select the existing-environment option and enter your actual project
path, then open **Configuration files**. Copy each configuration into its named
file under the displayed package directory, such as
`config/exported/my_model/`. Keep the four YAML files and `prepare.json` together;
copying only `model.yaml` is insufficient. Save `graph.json` as well if you want
to reopen the model in Studio later.

The guide still requires a valid ZIP-path value to generate its files, even for
manual copying. That value does not require an actual ZIP for the manual
commands below. **Do not run the generated `commands.sh` or the complete copied
flow without a ZIP:** both include a ZIP extraction command. After copying the
files, use the following manual preparation and training commands instead.

The package contains configuration, not datasets, pretrained weights or trained
checkpoints. The preparation command downloads supported missing resources and
preprocesses them **on the execution machine**. Before running it, align Hugging
Face caches with the project's resource directory, particularly for External HF:

```bash
cd /path/to/Legoizer
source .venv/bin/activate

export HF_HOME="$PWD/studio/backend/workspace/resources/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"

python -m studio.backend.executor.provision config/exported/my_model/prepare.json
```

After preparation succeeds, run training from the same project root. This
example uses a non-External-HF model and GPU 0:

```bash
python trainer.py \
  --model config/exported/my_model/model.yaml \
  --exp config/exported/my_model/exp.yaml \
  --data config/exported/my_model/data.yaml \
  --embed config/exported/my_model/embed.yaml \
  --cuda 0 \
  --seed 2023
```

Use `--cuda -1` for CPU, or the intended visible GPU index. External HF item
encoders use `prepared-model.yaml` created by preparation; their exported command
already selects this filename. Replace the example directory, model filename and
seed to match your files and intended run. Keep preparation and training in the
same shell so that the cache environment remains available. No Studio web service is needed
for these CLI steps. Some dataset-specific embedding matrices must be supplied
separately; see [data coverage and offline use](docs/DATA.md). Gated/private models
require authorized access and `HF_TOKEN` in the server environment; never save
credentials in exported configs or Git.

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

In Studio's **Training** settings, select the detected GPU entry you intend to
use, such as **CUDA:0**. The export guide accepts a manually entered GPU index
for the machine that will execute the exported commands.
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
use machine-wide process-name matching. It manages services launched through
`studio/run.sh`; it does not stop a separately launched uvicorn process. Logs are
in `studio/logs/`.

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

The **Official Benchmark** view uses the live external API in server mode. Pages
mode uses a bundled static snapshot of 828 configurations and their details,
labeled November 16 data; it does not store the actual capture time. Benchmark
results are separate from local runs. An unavailable live API does not prevent
server-mode graph editing or training.

## Publish or maintain a Pages site

Ordinary Pages visitors do not need these build steps. Maintainers can build with
the committed static catalogs using Node/npm only:

```bash
cd studio/frontend
npm ci
STUDIO_BASE_PATH=/Legoizer/ npm run build:pages
STUDIO_BASE_PATH=/Legoizer/ npm run preview:pages
```

Use the repository's actual Pages base path; use `/` for a root/custom-domain
site. The preview is local; it does not publish. On GitHub, select **Settings →
Pages → GitHub Actions**, then manually run **Actions → Studio GitHub Pages →
Run workflow**. A push alone does not publish with the current workflow.

| Build command | Runtime | Output |
| --- | --- | --- |
| `npm run build` | Frontend + Python backend | `studio/frontend/dist/` |
| `npm run build:pages` | Browser-only editor | `studio/frontend/dist-pages/` |

The directories are independent. Never substitute the Pages output for the
server's `dist/`. After changing Python components/templates, regenerate the
public catalog from the project root with `.venv/bin/python -m
studio.scripts.export_catalog`, then review and commit the generated catalog.
Static Benchmark refresh instructions are in [the mode guide](docs/STUDIO_MODES.md).

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
