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
-> Qwen Record Extractor：从正文和表格中抽取结构化科学记录
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
  --question "本地测试：抽取 PDF 和 CSV 中的科研指标。" `
  --files ../examples/perovskite_metrics.csv ../examples/demo_scientific_paper.pdf `
  --output-dir ../outputs
```

注意：`--allow-rule-fallback` 只用于本地测试，不能作为正式参赛结果。

## API

启动服务：

```powershell
pip install -r requirements.txt
uvicorn scidata_agent.api.main:app --reload --port 8000
```

接口：

```text
GET  /api/health
POST /api/discover
POST /api/analyze
GET  /api/tasks/{task_id}
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

1. 图表自校正回路：将校验疑点反馈给 Qwen-VL 二次读取，修正坐标轴/图例解析错误（赛题加分项）。
2. 缺失数据 / 单位不一致的自动识别与修正（赛题加分项）。
3. 人在回路修正闭环：让 `needs_review.csv` 的确认/修改能写回结果。
4. 增加网页、补充材料（Figshare/Zenodo 独立图片文件）解析能力。
5. 将 `ScientificRecord` 进一步通用化为 `entity/entity_type/attributes`。
6. 增加异步任务队列和持久化任务状态，方便前端调用。
7. 可交互前端（评分维度"作品演示、交互入口与交付完整度"）。
