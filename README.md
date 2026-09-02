# SciData Agent

SciData Agent 是面向“赛道二：数据场景 / 方向 A：科学数据查找、解析与整合”的后端 Data Agent。项目目标不是只做 PDF parser，而是把用户的科研问题转化为可执行的数据任务：发现数据源、解析多源材料、抽取结构化记录、保留证据链、做质量校验，并导出 CSV/JSON。

## 当前 Agent 工作流

```text
用户研究问题
-> Qwen Task Planner：理解任务、生成目标字段和动态 schema
-> Qwen Dynamic Schema Planner：根据用户问题生成本次任务专属的动态表结构和字段
-> Qwen Source Discovery：推荐论文、开放数据库、补充材料、表格、图像/图表等数据源
-> PDF/CSV/Excel Parser：解析本地上传文件（PDF 正文 + pdfplumber 结构化表格）
-> Figure Chart Branch：PyMuPDF 定位图表 caption 与图形区域并渲染 PNG -> Qwen-VL 分类与结构化读数（坐标轴/图例/数据点）-> 确定性校验（轴范围/序列/单位）
-> Qwen Dynamic Extractor：按照动态 schema 抽取论文画像、方法、数据集、实验设置、局限性等多表记录
-> 指标适配器：优先从动态抽取结果确定性生成指标记录；无可用数值时回退 Qwen Record Extractor
-> Schema Alignment / Normalizer：字段对齐、单位处理、重复合并
-> Provenance Tracker：记录 source_file/source_type/page/evidence_text
-> Rule + Qwen Quality Validator：检查证据覆盖、缺失字段、冲突值
-> Exporter：导出 result.csv、result.json、paper_survey.csv/json、source_discovery_plan.json、quality_report.json
```

如果用户没有上传文件，系统会运行“规划 + 数据源发现”模式，输出候选数据源、推荐关键词、目标数据类型和动态 schema。这一点用于对齐赛题中“用户只输入研究目标，Agent 自动查找相关论文/数据库/附件/图表”的要求。

## 环境变量

正式运行需要配置阿里云百炼或 DashScope 的 Qwen API Key：

```powershell
$env:DASHSCOPE_API_KEY="你的 API Key"
$env:QWEN_MODEL="qwen-plus"
$env:QWEN_VL_MODEL="qwen3-vl-30b-a3b-thinking"   # 图表/图像数据提取用的视觉模型（可选，有默认值）
```

项目也会自动读取 `backend/.env`。不要把 `.env` 提交或公开。
可从 `backend/.env.example` 复制完整配置；`QWEN_NODE_MAX_ATTEMPTS=2`
限制同一逻辑节点的总尝试次数，避免模型池耗尽后长时间盲重试。上传、并发、下载和
渲染等安全边界也集中列在该示例文件中。非本机部署时应设置 `SCIDATA_API_TOKEN`。
只有部署在会覆盖转发头的可信反向代理后，才可开启
`SCIDATA_TRUST_PROXY_HEADERS=true`。可信的单用户内部环境可在前端构建时设置相同的
`VITE_SCIDATA_API_TOKEN`，API 请求、图片预览和文件下载都会携带 Bearer Token；
但所有 `VITE_*` 值都会进入浏览器包，公开部署必须改用登录网关或 HttpOnly 会话，
不能把该构建变量当作服务端秘密。

如果任务日志出现 `HTTP 403`、`insufficient_quota` 或 `Free quota exhausted`，说明
DashScope 账户配额或计费状态不可用，不是本地依赖或服务启动故障。客户端会对这类账户级
错误快速失败，不再逐个轮询整个模型池；请更换有可用额度的 key、为账户充值，或在本地
测试时显式使用 `--allow-rule-fallback --legacy-runtime`。

## 命令行使用

进入后端目录：

```powershell
cd backend
```

只输入研究问题，生成数据源发现计划：

```powershell
python -m scidata_agent.cli `
  --question "我希望研究 Ia 型超新星光变曲线" `
  --discover-only `
  --output-dir ../outputs
```

上传本地文件，执行完整解析和抽取：

```powershell
python -m scidata_agent.cli `
  --question "上传论文和表格中提取材料、方法、PCE、稳定性、RMSE、吸收波长和来源证据。" `
  --files ../examples/perovskite_metrics.csv ../examples/demo_scientific_paper.pdf `
  --output-dir ../outputs `
  --max-pdf-pages 5
```

本地工具链测试可以显式开启 fallback：

```powershell
python -m scidata_agent.cli `
  --allow-rule-fallback `
  --legacy-runtime `
  --question "本地测试：抽取 PDF 和 CSV 中的科研指标。" `
  --files ../examples/perovskite_metrics.csv ../examples/demo_scientific_paper.pdf `
  --output-dir ../outputs
```

注意：`--allow-rule-fallback` 只用于本地测试，不能作为正式参赛结果。

## API

启动服务：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn scidata_agent.api.main:app --reload --port 8000
```

默认安装使用 `pdfplumber` 表格解析，体积更小、启动更快。只有明确需要本地
Table Transformer 推理时，才安装 `requirements-tatr.txt` 并设置
`USE_TABLE_TRANSFORMER=true`。真实模型测试也需要显式启用：
`RUN_TATR_TESTS=true pytest -m tatr`；默认测试不会加载 torch/TATR。

接口：

```text
GET  /api/health
GET  /api/tasks
POST /api/discover
POST /api/analyze
GET  /api/tasks/{task_id}
GET  /api/tasks/{task_id}/events
POST /api/tasks/{task_id}/cancel
POST /api/tasks/{task_id}/retry
POST /api/tasks/{task_id}/reviews/{record_id}
GET  /api/tasks/{task_id}/assets/{scope}/{asset_path}
GET  /api/tasks/{task_id}/export?format=csv
GET  /api/tasks/{task_id}/export?format=json
GET  /api/tasks/{task_id}/export?format=quality_report
GET  /api/tasks/{task_id}/export?format=source_discovery_plan
```

`POST /api/discover` 只需要 `research_question`，用于“研究问题 -> 数据源发现计划”。

`POST /api/analyze` 使用 multipart form：

- `research_question`：用户科研问题
- `files`：可选，一个或多个 PDF/CSV/Excel 文件
- `max_pdf_pages`：每个 PDF 最大解析页数，默认 8
- `enable_live_search`：是否在已上传文件之外继续联网查找多源资料
- `auto_download_sources`：是否自动下载选中的远程资料
- `max_auto_resources`：自动选择和获取的资料上限
- `reuse_dynamic_records_for_metrics`：复用动态抽取结果，避免重复调用指标抽取模型

完整响应结构、错误码、上传限制和资源 URL 规则见
[`backend/API_CONTRACT.md`](backend/API_CONTRACT.md)。

## Web 前端

前端位于 `frontend/`，使用 React、TypeScript、Vite 和 TanStack Query。
它是科研数据工作台，不是聊天页，当前包括：

- 新建完整分析或来源发现任务
- PDF/CSV/TSV/Excel 拖拽上传和参数设置
- 任务历史、实时节点进度和运行记录
- 来源目录、动态数据表、原始/清洗记录切换
- 字段证据抽屉、图表解析、质量报告和导出中心
- 人工复核结论持久化、失败任务重跑、排队/运行中任务协作式取消
- 最终 Markdown 调研报告在线预览

先启动后端：

```bash
cd backend
python -m uvicorn scidata_agent.api.main:app --reload --port 8000
```

再启动前端：

```bash
cd frontend
npm install
npm run dev
```

打开 `http://localhost:5173`。开发服务器默认把 `/api` 代理到
`http://localhost:8000`。若前后端分别部署，可在前端构建时设置
`VITE_API_BASE_URL`。任何 DashScope/Qwen 密钥都只能配置在后端，不能写入
`VITE_*` 环境变量或浏览器存储。

## 输出文件

每次任务会在时间戳目录下生成：

- `result.csv`：结构化科研数据表（含 PDF 表格提取的记录，source_type = pdf_table）
- `result.json`：任务计划、发现计划、记录、来源和质量报告的完整 JSON
- `dynamic_schema.json`：由 Qwen 根据用户问题生成的动态抽取 schema，包含本次任务应抽取的动态表和字段
- `dynamic_records.json`：按照动态 schema 抽取出的全部动态记录
- `tables/`：按动态表名导出的多个 CSV，例如 `method_details.csv`、`dataset_usage.csv`、`experiment_results.csv`
- `summary.json`：给前端或用户快速读取的任务摘要
- `final_report.md`：给人看的调研报告，概述任务、动态 schema、论文汇总、动态表预览和质量信息
- `paper_survey.csv`：按论文聚合的调研表，包含论文题目、作者、arXiv 链接、PDF 链接、方法、数据集/对象、指标和证据样例
- `paper_survey.json`：`paper_survey.csv` 的 JSON 版本，便于前端读取
- `source_discovery_plan.json`：候选论文/数据库/附件/图像/表格来源和动态 schema
- `processing_log.json`：Agent 节点和处理日志
- `quality_report.json`：证据覆盖、字段覆盖、冲突、警告和错误（含图表校验合并结果）
- `figures/`：从 PDF 定位并渲染的图表 PNG
- `chart_extractions.json`：Qwen-VL 提取的图表坐标轴、图例和数据点（含来源溯源）
- `chart_data/`：每张图表的长表 CSV（`chart_data_index.csv` 为索引）
- `chart_validation_report.json`：图表提取的确定性校验（轴范围/序列/单位一致性，needs_review 标记）
- `agent_monitor.jsonl`：逐步运行监控日志，记录每个 Agent 节点的开始、结束、耗时和关键输出摘要

## 测试

推荐使用项目 conda 环境：

```powershell
conda activate scidata-agent
cd backend
python run_tests.py
```

当前测试覆盖：

- Mock Qwen 完整 Agent pipeline
- Qwen 动态 schema planning、动态记录抽取和动态多表导出
- 只输入研究问题的数据源发现模式
- 未配置 Qwen Key 时正式模式失败
- fallback 模式显式标记
- 质量报告、证据检查和冲突检测
- 不同领域的通用 source discovery fallback
- 图表定位、VL 提取与校验
- PDF 结构化表格提取（pdfplumber lines/text 双策略 + 质量过滤）

## 下一步开发重点

1. 图表自校正回路：将校验疑点反馈给 Qwen-VL 二次读取，修正坐标轴/图例解析错误。
2. 让人工复核的“需要修改”结论支持字段级编辑，并生成新的结果版本。
3. 增加网页正文、Figshare/Zenodo 独立图片和更多补充材料格式的解析能力。
4. 将本地线程任务执行器替换为可横向扩展的持久化队列。
