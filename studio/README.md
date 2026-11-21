# Legoizer — Visual Programming for Recommendation Models

![status](https://img.shields.io/badge/status-working-blue) ![backend](https://img.shields.io/badge/backend-FastAPI-green) ![frontend](https://img.shields.io/badge/frontend-React%20%2B%20React%20Flow-blue)

**Drag → Connect → Configure → Validate → Compile → Train → Evaluate**

Legoizer is an interactive visual programming environment built **on top of the real
[Legommenders](https://github.com/Jyonn/Legommenders) code base**. Every component shown in the palette is discovered from the
actual Python source (`loader.class_hub.ClassHub`), the shipped YAML configurations and
`utils.metrics.MetricPool`. The generated model file is a real Legommenders YAML configuration,
verified through `trainer.get_configurations()` and the real `config_class` constructors before it
is offered for download or training.

Studio uses the original Legommenders training and evaluation commands. The shared
Trainer also supports gradient accumulation for Studio and CLI runs.

---

## 1. Contents

```
studio/
├── frontend/                  React 18 + TypeScript + Vite + Tailwind + @xyflow/react + Zustand
│   └── src/
│       ├── api/               single typed HTTP client
│       ├── canvas/            React Flow editor
│       ├── components/        Toolbar, Library, Properties, Bottom panels, dialogs, toasts
│       │   └── panels/        Console, YAML, Training, Evaluation, Comparison, Graph JSON
│       ├── hooks/             keyboard shortcuts
│       ├── nodes/             typed card node with semantic ports
│       ├── store/             Zustand graph store (undo/redo, persistence)
│       ├── types/             TypeScript mirrors of the Pydantic schemas
│       └── utils/             icons / colours / formatting
├── backend/
│   ├── api/routes.py          FastAPI endpoints
│   ├── registry/              dynamic component registry + curated semantic overlay
│   ├── schemas/               Pydantic v2 schemas (graph, components, runs)
│   ├── validator/validator.py graph / port / dimension / task / config checks
│   ├── compiler/              Graph JSON → Legommenders YAML + real-loader verification
│   ├── templates/templates.py templates built by loading the real YAML files
│   ├── executor/executor.py   subprocess-based training/evaluation, logs, run records
│   └── tests/                 backend + integration tests
├── scripts/env.sh             shared runtime bootstrap
├── run_backend.sh             serve API and the built frontend
├── run_frontend.sh            start only the Vite dev server
├── run.sh                     start the built app (or both in development)
├── stop.sh                    stop both
└── backend/requirements.txt   Studio runtime dependencies
```

---

## 2. Installation and startup

See the root [README](../README.md) for complete CPU, NVIDIA GPU, WSL2,
and remote-server instructions. Python 3.11 and Node.js 20 or newer are required
for installation. A maintained Node.js LTS release is recommended.

```bash
bash studio/setup.sh --torch cpu --dev
bash studio/run.sh
```

The default URL is `http://127.0.0.1:8000/`. The backend serves the compiled
frontend and `/api` from this one port. No `.data` file is required.
The first training run downloads missing supported dataset/model resources.

For frontend development, use `STUDIO_DEV=1 bash studio/run.sh`; Vite opens on
`http://127.0.0.1:5173/` and proxies `/api` to the backend. Stop with Ctrl+C or
`bash studio/stop.sh` from another terminal in this checkout. Standalone
`run_backend.sh` and `run_frontend.sh` can be run in separate terminals and
stopped with Ctrl+C in those terminals. Cancel active training before stopping.

---

## 4. First-run workflow

1. Click **Open NAML Demo**. It opens the real NAML graph with a smaller MIND dataset
   and training settings suitable for a short demonstration.
2. Drag a block from **Component Library** to the canvas. Connect its output handle to
   a compatible input handle. To try a model variation, replace the Additive Attention
   user encoder with GRU.
3. Click **Validate** and **Generate Model** to see the resulting Legommenders YAML.
4. In **Training**, set **Accumulate batches** to 3 if you want three mini-batches
   per optimizer update. Click **Train** to prepare resources and start training.
5. Open **Evaluation** to see the metrics emitted by the real trainer.

> **Truthfulness note.** The official `config/model/lstur.yaml` uses the `CNNCat` Item Encoder,
> while `README.md` lists LSTUR as *CNN + GRU + Dot*. A graph using plain `CNN + GRU + Dot`
> compiles to a **valid** Legommenders configuration named `LSTUR`, but the compiler adds an
> explicit note saying it is not a byte-for-byte copy of the shipped YAML. Combinations that do
> not match the README model table are always labelled **Custom Model**.

---

## 5. How to use the editor

| Action | How |
|---|---|
| Add a block | Drag an item from the left library onto the canvas (or double-click it) |
| Move / pan / zoom | Drag a node; drag empty canvas; scroll to zoom |
| Connect | Drag from a right-hand (output) handle to a left-hand (input) handle |
| Port typing | `isValidConnection` checks semantic port types; illegal links are refused with a toast. Example: `Dataset` cannot feed an `ItemContent` input |
| Select / configure | Click a node → right **Properties** panel; edits update the node summary, Graph JSON and compiled config immediately |
| Restore defaults | **Defaults** button in the Properties panel |
| Delete / duplicate | `Delete` key, right-click menu, or the Properties buttons |
| Delete an edge | Click the line → `Delete` |
| Replace a module | Select node → **Replace** / right-click → Replace Component |
| Add compatible module | Select node → **Compatible**; backend returns only port-compatible components with reasons |
| Auto layout / Fit view | Buttons on the canvas top-left, or the right-click menu |
| Save / restore | `Ctrl/Cmd+S` writes to `localStorage` **and** the server workspace; a saved graph is restored on refresh. Server projects are listed in the *Graph JSON* tab |
| Import / export | Toolbar *Import* (`.json` Graph JSON or `.yaml` model config), *Export* (Graph JSON), and *Download* in the YAML tab |
| Tasks | Switch `matching`/`ranking` in the Properties panel footer or the Training panel |
| Themes | Moon/Sun button in the Toolbar |
| Keyboard | `Ctrl/Cmd+S`, `Ctrl/Cmd+Z`, `Ctrl/Cmd+Shift+Z`, `Ctrl/Cmd+Y`, `Delete`, `Ctrl/Cmd+A`, `Esc` |
| Validation errors | Console tab; clicking an issue centres the corresponding node |

Component categories: **Dataset** (blue), **Feature Components** (teal, conceptual),
**Item Encoder** (green), **User/Behavior Encoder** (orange), **Predictor** (purple),
**Training** (pink).

### Conceptual vs executable blocks

The *Feature Components* category (User ID, Item ID, Item Text, Category, User History,
Item/User Embedding) contains **annotations only**: they are marked `CONCEPTUAL` in the
library, shown with dashed borders, and excluded from the generated YAML. They document the
real `data.item.inputs` / `column_map` fields. Training blocks (Matching Task, Ranking Task,
Negative Sampling, BCE Loss, Cross Entropy) *are* executable: they control
`config.use_neg_sampling`, `config.neg_count` and the loss chosen by
`model/legommender.py`. Evaluation metrics are selected under **Training** and map to real `utils.metrics.MetricPool` classes. Existing graphs migrate evaluation blocks into the training settings.

---

## 6. Architecture

```
Graph JSON  ──► Graph Validator ──► Component Registry ──► Model Compiler ──► Legommenders YAML
   (frontend)      (Pydantic)          (ClassHub + YAML)        │                  │
                                                                 └──► trainer.get_configurations()
                                                                      ClassHub + config_class
                                                                      instantiation (verification)
                                                                              │
                                                    trainer.py / tester.py ◄── subprocess argv
                                                                              │
                                          real stdout logs + <signature>.csv  ──► Evaluation UI
```

### Component Registry (`GET /api/components`)

* `loader.class_hub.ClassHub` discovers all `BaseOperator` / `BasePredictor` subclasses.
* `inspect.signature` over each `config_class.__init__` MRO yields real parameter names,
  annotations and defaults.
* `refconfig.jyonnie_config.parse` reads every `config/model/**/*.yaml`, resolves `$$import`
  and learns the real `meta.*` values, roles and config keys (including `${var:default}$`
  defaults and nested `inputer_config.*`).
* `utils.metrics.MetricPool.metric_dict` provides real evaluation metrics.
* `config/data/*.yaml` + `config/recbench/*.yaml` are scanned for real trainer data configs.
* `studio/backend/registry/metadata.py` adds only prose, icons, port contracts and numeric
  ranges. **The frontend never sees or depends on a Python import path.**

A newly added `model/operators/my_operator.py` subclass appears in the palette automatically
(with an "AUTO" badge); add an entry to `COMPONENT_NOTES` for a nicer description/parameters.

### Graph Validator (`POST /api/graph/validate`)

Connection checks (port type compatibility, dangling/duplicate edges, self loops, cycles,
orphans, missing required inputs), configuration checks (Pydantic + registry parameter
type/range/choices/required), structural checks (exactly one dataset / user encoder /
predictor, item encoder required for matching), task checks (reusing the framework's
`allow_matching` / `allow_ranking` flags used by `LegoConfig.build_components`), and dimension
checks (all operators share one `LegoConfig.hidden_size`). Dimensions that depend on the loaded
UniTok tables are reported as **runtime validation** instead of being guessed.

Every issue has `message`, `reason`, `fix`, `node_id` (click to locate).

### Model Compiler (`POST /api/graph/compile`)

Emits only keys declared by `LegoConfig.__init__` and by real `config_class` signatures.
No `$$import` and no `${...}` placeholders: the generated file is self-contained and loadable
by `trainer.py`. Original template files are never written. The compiler also generates an
`exp.yaml` (epochs / lr / batch_size / metrics / patience) inside the server workspace.

### Executor (`POST /api/train`, `POST /api/evaluate`)

* `subprocess.Popen([sys.executable, "-u", "trainer.py", "--data", ..., "--model", ...])`
  — **argument array, no shell, no user-supplied command strings**.
* Only server-controlled config paths from allow-lists are accepted.
* Preflight resolves the real data config with `get_configurations()` and checks every UniTok
  path, every required `.npy` embedding, CUDA availability and the trainer script.
* Logs are streamed from the real process (`GET /api/train/{id}/logs`, SSE
  `/api/train/{id}/stream`), status is `pending → running → completed | failed | cancelled`.
* Test metrics are parsed from the real `tester.py` output and
  `checkpoints/<data>/<model>/<signature>.csv`; loss curves come from the real log file
  (`[epoch N] step M, loss X`). **Nothing is fabricated.**

---

## 7. HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness |
| GET | `/api/components` | full dynamic registry |
| GET | `/api/components/{id}/compatible?direction=upstream\|downstream` | port-compatible components |
| GET | `/api/templates`, `/api/templates/{id}` | real-YAML templates |
| POST | `/api/graph/validate` | validation report |
| POST | `/api/graph/compile` | YAML + verification notes |
| POST | `/api/graph/import` | import Graph JSON or Legommenders YAML |
| GET | `/api/environment` | torch / CUDA / deps / dataset / embedding status |
| POST | `/api/diagnostics/verify` | real GPU-free config + component instantiation check |
| POST | `/api/train` | validate → preflight → compile → real `trainer.py` |
| GET | `/api/train/{id}/status`, `/logs?after=N`, `/stream` | status / polling / SSE |
| POST | `/api/train/{id}/cancel` | terminate the process |
| POST | `/api/evaluate` | real `tester.py` run |
| GET | `/api/runs`, `/api/runs/{id}`, `/model.yaml`, `/graph.json` | experiment history + downloads |
| POST | `/api/graphs/save`, GET `/api/graphs`, `/api/graphs/{id}` | project persistence |
| GET | `/api/reference/{path}` | whitelisted shipped model YAML only |

---

## 8. Model templates

Templates are **generated at runtime from the real YAML files** by calling
`trainer.get_configurations()`; they are not hand-written JSON. Available and verified:

| Template | Task | Real YAML | Graph |
|---|---|---|---|
| NAML | matching | `config/model/naml.yaml` | CNN + Ada + Dot |
| NRMS | matching | `config/model/nrms.yaml` | Attention + Attention + Dot |
| LSTUR | matching | `config/model/lstur.yaml` | CNNCat + GRU + Dot |
| MINER | matching | `config/model/miner.yaml` | Transformer + PolyAttention + MINER |
| PNN | ranking | `config/model/pnn_id.yaml` | Pooling + PNN |
| DeepFM | ranking | `config/model/deepfm_id.yaml` | Pooling + DeepFM |
| DCN | ranking | `config/model/dcn_id.yaml` | Pooling + DCN |
| DIN | ranking | `config/model/din_id.yaml` | NullConcat + DIN |
| AutoInt, DCNv2, MaskNet, GDCN, FinalMLP | ranking | `config/model/*_id.yaml` | Pooling + predictor |

Ranking templates deliberately have **no** Item Encoder (`use_item_content: false`,
ID-based), matching the shipped `*_id.yaml` files.

### Adding a new template

1. Put a real YAML file in `config/model/` (the original project layout).
2. Add one entry to `studio/backend/templates/templates.py::TEMPLATE_DEFS` with `id`, `name`,
   `yaml`, `task`, `description`, `paper`.
3. Optionally add an archetype to `studio/backend/registry/metadata.py::MODEL_ARCHETYPES`
   so the compiler labels the combination with its paper name.
4. Restart the backend (`GET /api/templates` is rebuilt at startup).

### Adding a new component

1. Add `model/operators/foo_operator.py` (subclass of `BaseOperator`) or
   `model/predictors/foo_predictor.py` (subclass of `BasePredictor`) with a `config_class`.
2. Add a `config/model/common/operators/foo.yaml` or `predictors/foo.yaml` that references it
   via `meta.item/user/predictor: Foo` — this makes it `verified`.
3. Optionally add prose / parameter descriptions in
   `studio/backend/registry/metadata.py::COMPONENT_NOTES` / `COMMON_PARAM_NOTES`.
4. Restart the backend. `ClassHub` discovers it, `inspect.signature` publishes its real
   parameters and the frontend palette updates automatically.

---

## 9. Running real training

Open a template, select a CPU or CUDA device in Training, and click **Train**.
Compatible prepared tables and weights are reused. Missing resources are
prepared in a background process with progress in the Training log. **Cancel
run** stops preparation or training. The optional `.data` file only overrides
raw-data locations; see [the data guide](../docs/DATA.md).

**MIND Demo** samples 650 real MIND-small users with a fixed seed. It still
requires the original MIND-small downloads and the selected token resources.
GloVe's initial archive is approximately 862 MB. A demo is a smaller training
workload, not a bundled offline dataset or an experimental benchmark.

Gated Hugging Face models need your own authorized read token. Use local
loopback access, an SSH tunnel, or configured HTTPS to submit it. Tokens are
not written to graph JSON or run records. See the root README for the tunnel
command and [deployment settings](../docs/DEVELOPMENT.md).

Choose metrics and the dev metric under Training. `accumulate_batch` maps to
`exp.policy.accumulate_batch`; the final partial accumulation group is applied.
The Properties and Training panels edit the same settings.

---

## 10. Tests

```bash
# Backend + integration
cd /path/to/Legoizer
.venv/bin/python -m pytest studio/backend/tests -q

# Frontend tests
cd studio/frontend
npm test
```

A browser-level **End-to-End Acceptance Test** is also provided in
`studio/tests/browser_e2e.cjs`. It drives a real Chromium through
*Welcome → NAML → replace Additive Attention with GRU → Validate → Generate Model → LSTUR
YAML → Train*, and asserts that the trainer completes and emits real GAUC, MRR and NDCG. Run it while the Studio is running:

```bash
export PLAYWRIGHT_CORE=/path/to/playwright-core
export CHROME_PATH=/path/to/chrome        # optional
STUDIO_URL=http://127.0.0.1:5173 node studio/tests/browser_e2e.cjs
```

Run acceptance tests against your current build; they may download resources and start real training.

Coverage:

* **Frontend store**: add node, move node, valid/invalid/duplicate/self connections, delete
  node, delete edge, parameter modification and `hidden_size` sync, reset defaults, duplicate,
  Graph JSON serialisation, save/load, undo/redo, model morphing (Ada → GRU), auto layout.
* **Backend**: registry loads real classes, real parameter defaults, meta values resolvable by
  `ClassHub`, conceptual flags, port compatibility, dataset files; validator rejects unknown
  components, port mismatches, orphans, cycles, bad parameter types/ranges, dimension
  mismatches, DIN-under-matching; compiler emits loadable YAML with no invented keys,
  preserves original templates, YAML import round-trip, NAML→LSTUR morph, Custom Model naming.
* **API**: all endpoints, whitelisted reference YAML, preflight honesty, save/load.
* **Integration**: NAML template → validate → compile → `trainer.get_configurations()` →
  `ClassHub` → real `config_class` instantiation → morph to LSTUR → re-verify; original CLI
  config loading still works; training environment recorded without fabricated metrics.

The browser test also drags an xMIND block from the library and submits a real run
with `accumulate_batch=3`.

---

## 11. Security & isolation

* Pydantic v2 validates every request body (`extra="forbid"`).
* No `shell=True`; training uses a `subprocess` argument array with a fixed script name.
* Data / embed / model config paths are resolved against server-controlled allow-lists;
  `/api/reference/{path}` only serves `config/model/**/*.yaml`.
* Generated YAML, experiment configs, run records and logs are written only under
  `studio/backend/workspace/`; shipped `config/` files are read-only.
* Run processes are tracked, cancellable, and their status survives a backend restart
  (a previously `running` record is marked `failed` rather than shown as successful).
* The original Legommenders CLI is never wrapped or replaced.

---

## 12. Known limitations

* Provider access, download size and GPU memory determine whether a first run
  can prepare a given checkpoint. The log shows the failing preparation step.
* The library excludes LM base classes and two legacy semantic prototypes that
  cannot construct models with the current UniTok API.
* The upstream Fastformer YAML references a missing import directory.
  The Fastformer components are available individually in the library.
* Validation curves only appear when `exp.policy.check_interval` logging produced
  `[epoch n] metric value` lines; otherwise the chart shows an explicit empty state.
* `Ctrl/Cmd+A` selects the first node and reports the count; visual multi-select uses
  `Shift+drag` (React Flow's own selection model).

### Expanded Model Canvas

Complete recommendation graphs open in **Data Flow** view. The upper branch
shows three representative history items (with an ellipsis before the final
item), their shared item encoder instances and embeddings, and the user encoder
and embedding. Matching displays one positive and up to two negative candidate
branches, using an ellipsis when `neg_count > 2`; ranking displays one candidate.
Each candidate has its own encoder, embedding, predictor and score leading to
the configured task. Repeated encoder/predictor boxes share configuration and
weights; clicking any instance opens the original component properties.
Architectures without an item encoder show dataset/precomputed item embeddings.
The dataset appears as a data-source context card; metadata/configuration links remain in Config view. Feature and evaluation nodes retain their configured connections.

Use **Data Flow → Config** to edit component connections, positions or replace
components using the context menu. The expanded view is a display projection;
it does not duplicate operators in graph JSON or emitted YAML. Incomplete graphs
continue to open in the editable configuration view.

Data Flow opens in canvas focus mode to make the full chain readable. Use **Exit focus** to restore the component library and properties, or click a configured encoder/task to edit it. Shared user embeddings use a dedicated pink routing gutter.


### Node sizing and Config layout

Data Flow and Config use the same node card shell, category colors and operator
icons. Long titles expand the card width up to 360 px, then wrap; descriptions,
parameter values and port labels wrap and expand the height. Layout uses the
measured card dimensions, so growing cards trigger spacing and route updates.

Config Auto Layout derives layers from the graph's real edges. Content-based
models keep their dataset → item → user → predictor → task chain and skip
connections. ID-based models retain their configured feature/history routes
without inserting a content encoder. Disconnected nodes remain disconnected.
Orthogonal routes avoid card bounds, including direct dataset/task links that
skip intermediate operators. Nodes can still be dragged; Auto Layout restores
the calculated layout. Focus canvas is available in both views.

Regression fixtures in `frontend/src/canvas/fixtures/modelGraphs.json` capture
all 14 available server template graphs. Tests check every actual edge, topology
preservation, and obstacle avoidance with variable card dimensions.

The canvas overview and its dock have been removed. The model canvas uses the full available width.

### Interface languages

Studio supports **English** and **简体中文**, selectable in the toolbar or welcome
screen. It remembers the choice in this browser and otherwise defaults to English on the first visit. Chinese parameter labels display only
the translated name; model, dataset and metric names remain English. Language
switching updates labels and canvas measurements without changing model semantics,
configuration keys, custom node labels or training state. Raw process logs and
unknown third-party diagnostics remain in their original language.

See `frontend/src/i18n/README.md` for translation conventions and adding languages.
The bilingual browser test is `node studio/tests/i18n_e2e.cjs` (set
`PLAYWRIGHT_CORE`, `CHROME_PATH` and optionally `STUDIO_URL`).


### Official Benchmark results

Open **Benchmark** in the top toolbar or welcome screen.
This read-only view loads all official configurations, supports dataset and
model/dataset/LM filtering (space-separated terms), metric sorting, all metrics as mean ± standard deviation columns, per-seed results, and downloading
configuration JSON. Select a dataset to see a metric comparison chart.

Studio proxies the public API through `GET /api/benchmark/evaluations` and
`GET /api/benchmark/evaluations/{signature}`. The source is
https://lego.qijiong.work/evaluations/; its contract was verified against
https://github.com/legommenders/lego-backend (`backend/urls.py`,
`evaluation/views.py`, `evaluation/models.py`). Upstream returns SmartDjango's
`code/body` envelope. Lists use `?page=N&page_size=100` (maximum 100); detail
uses `/evaluations/{signature}` without a trailing slash and includes
`configuration`. The adapter reads all pages before publishing an index and
caches successful indexes in memory for 5 minutes. Network failures appear as
retryable errors; no synthetic or partial results are substituted.

Means and sample standard deviations (`ddof=1`, matching upstream's export)
include completed seeds with finite numeric metrics only. A single seed has
no standard deviation. Missing results remain unavailable. Official data stays
separate from local training runs and never changes the current model graph.

Benchmark is a standalone page at `/#/benchmark`. The top toolbar opens it;
Back to Studio returns to the editor with the bottom execution panel preserved.


### Local runs and Benchmark baselines

The bottom Comparison panel is dedicated to model selection. Select completed
local runs, then choose **View comparison** to open a wide right-side results
drawer. Closing it preserves selections and the baseline. To add official results, choose
**Add Benchmark comparison**. The drawer lists official results for the matched
dataset, with model/LM search. Local runs and official configurations share a
six-column comparison limit. Select an official baseline to display local score
minus official mean for matching datasets only. Official cells show mean ±
sample standard deviation, with per-metric sample counts on hover; missing values
remain unavailable. Details show parameters, configurations and seed results, then return to the
comparison drawer when closed. The former expand/restore mode has been removed.
Official records are read-only and are never added to local training history.

Dataset matching uses explicit configuration paths and aliases. MIND Demo,
ONCE-MIND and xMIND variants are excluded from full MIND baselines. Unknown legacy
datasets require an explicit manual match, labelled in the table. Dataset names
alone do not establish identical splits, candidate sets or evaluation protocols;
users must review the local parameters and official configuration in Details.
Deltas use raw score units and do not imply statistical significance.

`utils/benchmarkComparison.test.ts` covers dataset matching, exclusions, metric
ordering, missing values and delta arithmetic. The optional browser acceptance
test `studio/tests/benchmark_comparison_e2e.cjs` uses browser-only local-run fixtures
and the real official API; it never writes training records.
