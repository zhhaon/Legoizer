# Studio: local editor and server modes / 本地编辑与服务器模式

One frontend supports two explicit build modes. Server mode is the default;
backend failure does not silently switch modes or replace the component catalog.
同一前端支持两种明确的构建模式。默认使用服务器模式，后端断连不会自动切换目录或模式。

## Server / 服务器模式

Use the existing backend and `npm run build`. `VITE_STUDIO_API_URL` optionally
sets the API base (including `/api`); otherwise `/api` is used. Remote backends
need HTTPS, authentication appropriate for deployment, and the exact frontend
origin in `STUDIO_CORS_ORIGINS`. There is no runtime remote-backend connection UI.
继续使用现有后端和 `npm run build`。可通过 `VITE_STUDIO_API_URL` 设置 API 根地址
（包含 `/api`），默认同源 `/api`。跨站部署需配置 HTTPS、部署认证和 CORS。
本版不提供网页运行时连接远程后端的界面。

## Local editor / 纯前端模式

From the project root, using the project's Python environment:
在项目根目录中使用项目 Python 环境生成公共静态目录：

```bash
.venv/bin/python -m studio.scripts.export_catalog
cd studio/frontend
npm ci
STUDIO_BASE_PATH=/Legoizer/ npm run build:pages
```

Publish `studio/frontend/dist-pages` to GitHub Pages. For a root/custom domain use `/`
as the base path. The manual **Studio GitHub Pages** workflow builds and deploys
the committed catalog; it does not start on push. Configure Pages to use GitHub
Actions, then dispatch the workflow when ready. No publication is needed for local use.
发布 `dist-pages`；根域名部署使用 `/`。仓库提供手动触发的 GitHub Actions 工作流，使用已提交
的静态目录，不会因 push 自动发布。需将 Pages 来源设为 GitHub Actions。

Regenerate and commit `public/catalog.json` after changing Python components,
configuration schemas, templates or model defaults. It contains only public
metadata/configuration, never local environment paths, tokens or training records.
The recorded source revision identifies the catalog baseline. Local mode never
requests `/api`: it uses static catalog/templates, browser validation/compiler,
local project storage and file import/export. Training/evaluation/resource controls
explain that execution happens on the user's server. Benchmark uses a bundled static snapshot in local mode.
修改 Python 组件、配置格式、模板或默认值后，请重新生成并提交静态目录。目录只包含公共配置，
不含令牌、训练历史或本机环境信息。纯前端模式不请求 `/api`，支持本地保存、导入、校验与导出。
训练、评估和资源准备需后端；Benchmark 在纯前端模式使用打包的静态快照。浏览器校验不会声称完成 Python 实例化检查。

## Portable packages / 可迁移配置包

Open **YAML Preview** or click the mode badge. Set a relative directory, generate
the package, then copy any file/commands or download individual files/a ZIP.
The same interface is available in server mode, where configs are resolved by Python.
打开 YAML Preview 或点击模式提示，设置相对目录后生成配置包，可复制或下载单个文件，也可下载 ZIP。
服务器模式使用相同入口，由 Python 解析配置。

Extract the ZIP into the root of a matching Legoizer checkout. The ZIP includes
`config/exported/<name>/{model,exp,data,embed}.yaml`, `prepare.json`, `graph.json`
and bilingual README content selected by the UI language. Copy users must save
these files under the displayed directory. Run the displayed resource preparation
command before training. Preparation can download resources and replaces the
package's data/embed YAML with prepared configs. External HF preparation also
writes `prepared-model.yaml`, which its training command uses. No automatic
training or download occurs when exporting. Install the project dependencies and
provide the raw datasets/authorized HF_TOKEN as needed. All paths are relative to
the project root. Set device and seed in Training before export. Auto device exports
GPU 0; select CPU explicitly for CPU servers. The command retains the seed.
将 ZIP 解压到匹配版本项目根目录。复制方式需按展示目录保存所有文件。先执行资源准备命令，
再训练；准备步骤可能下载资源并更新包内 data/embed 配置。External HF 会生成 prepared-model.yaml。
导出操作本身不会下载或训练。目标服务器需要安装依赖、准备原始数据，并在必要时设置 HF_TOKEN。
设备和种子沿用 Training 设置；自动设备导出为 GPU 0，CPU 服务器请明确选择 CPU。

Local YAML import accepts resolved model YAML; unresolved imports/variables are
rejected. Graph JSON preserves all training/dataset settings and is preferred for
moving projects between modes. Browser-known columns come from the static config;
additional columns present only in server UniTok tables require server mode or a
refreshed catalog. Static checks cannot verify files, GPU memory or Python construction.
本地 YAML 导入只接受已展开的模型配置。跨模式迁移推荐 Graph JSON，可保留完整训练和数据设置。
静态目录无法发现服务器 UniTok 表中的额外字段，也不能验证本机文件、GPU 显存或 Python 构造。

## Separate build outputs / 分离构建产物

`npm run build` explicitly selects server mode, uses root asset URLs and writes
`dist/`, which the Python backend serves. `npm run build:pages` explicitly selects
local mode and writes `dist-pages/`. Pages builds never clean or overwrite `dist/`.
Use `STUDIO_BASE_PATH=/Legoizer/ npm run preview:pages` to preview a Pages build
with the same base path used when building. Mode selection overrides a leftover
`VITE_STUDIO_MODE` environment variable for these two build scripts.

`npm run build` 明确使用前后端模式、根路径资源地址，输出到后端托管的 `dist/`。
`npm run build:pages` 明确使用纯前端模式，输出到 `dist-pages/`，不会清空或覆盖服务器产物。
预览 Pages 时请使用与构建一致的子路径，例如
`STUDIO_BASE_PATH=/Legoizer/ npm run preview:pages`。两个构建脚本均显式指定模式，
不受遗留的 `VITE_STUDIO_MODE` 环境变量影响。

## Static Benchmark / 静态 Benchmark

Local mode reads `public/benchmark.json` from the Pages base path, including all
configuration details. Filtering, sorting, charts and JSON downloads work without
backend calls. The display label is November 16 data; `snapshot_date` is the
requested display date; the static file does not retain capture timestamps.
This is not a reconstructed historical API response.
Server mode continues to use the live API. Refresh explicitly from the root:

```bash
.venv/bin/python -m studio.scripts.export_benchmark --display-date 2026-11-16
```

纯前端从 Pages 子路径加载静态列表及全部配置详情，保留筛选、排序、图表和下载。
界面按要求标注“11月16日数据”；文件不保留实际抓取时间。
服务器模式继续使用实时 API。更新静态快照后需重新构建 Pages。
