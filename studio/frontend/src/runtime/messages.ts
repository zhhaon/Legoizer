import i18n from '@/i18n';

const en = {
  local: 'Local editor · export and run on your server', server: 'Server mode',
  staticCheck: 'Static checks only. Python, resources and GPU must be checked on your server.',
  serverOnly: 'This operation needs a backend. Export the configuration package to run on your server.',
  export: 'Run configuration package', zip: 'Download ZIP', copy: 'Copy selected', download: 'Download selected',
  folder: 'Directory under the project root', command: 'Commands', prepare: 'Prepare resources first; this may download data and weights.',
  requirements: 'Use the matching Legoizer version. Run from the project root. Data, weights and access tokens are not included. Set HF_TOKEN in your server environment for gated models. Adjust data/embedding paths and GPU index for your server.',
  version: 'Catalog revision', error: 'Cannot export', invalid: 'Invalid graph',
  structure: 'Check required components, connections, parameter types and dimensions.',
  imports: 'Local YAML import requires a resolved model mapping without imports or variables.',
  copied: 'Copied', path: 'Use a relative directory with letters, digits, underscores, hyphens and slashes; no parent traversal.',
  unavailable: 'This dataset/tokenizer combination is not in the static catalog. Rebuild the catalog with the matching project version.',
  storage: 'Saved in this browser',
};
const zh: typeof en = {
  local: '本地编辑模式 · 导出后在自己的服务器运行', server: '服务器模式',
  staticCheck: '仅完成静态检查。Python、资源和 GPU 需要在自己的服务器上验证。',
  serverOnly: '此操作需要后端。请导出配置包，在自己的服务器上运行。',
  export: '运行配置包', zip: '下载 ZIP', copy: '复制当前内容', download: '下载当前文件',
  folder: '项目根目录下的配置目录', command: '运行命令', prepare: '请先准备资源；此步骤可能下载数据和权重。',
  requirements: '请使用匹配版本的 Legoizer，在项目根目录执行命令。配置包不含数据、权重或访问令牌。受限模型需要在服务器环境中设置 HF_TOKEN。请根据服务器情况调整数据、embedding 路径和 GPU 编号。',
  version: '组件目录版本', error: '无法导出', invalid: '模型图无效',
  structure: '请检查必要组件、连接、参数类型和维度。',
  imports: '本地 YAML 导入需要已展开的模型配置，不支持导入引用或未解析变量。',
  copied: '已复制', path: '请输入由字母、数字、下划线、连字符和斜线构成的相对目录，不允许上级目录跳转。',
  unavailable: '静态目录中没有此数据集与 tokenizer 的组合，请使用匹配项目版本重新构建目录。',
  storage: '已保存到此浏览器',
};
export function rt(key: keyof typeof en) { return (i18n.resolvedLanguage?.startsWith('zh') ? zh : en)[key]; }
