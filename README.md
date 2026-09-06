# SciData Agent

SciData Agent 是一个面向科研人员的数据整理与证据追溯工作台。输入研究问题，或上传论文和数据文件后，系统会检索相关资料、解析文献与表格、提取结构化信息，并把每条结果关联到可核查的来源证据。

它适合用于前期调研、论文信息整理、实验数据归档和跨来源结果对比。系统输出用于辅助科研工作；关键结论仍应由使用者回到原始来源进行确认。

## 你可以用它做什么

- 根据研究问题发现论文、开放数据集和其他公开资料。
- 从 PDF、CSV、TSV、XLSX/XLS 中提取材料、方法、指标、实验条件等信息。
- 按本次研究问题自动组织数据表和字段，不要求先建立固定模板。
- 查看每条记录的来源、页码、原文证据、抽取方式和置信度。
- 发现缺失字段、证据不足、低置信度和跨来源冲突，并交给人工复核。
- 下载 CSV、JSON、调研报告和质量报告，供后续分析或归档。

## 一次任务如何完成

```text
研究问题或本地文件
        ↓
规划数据结构与检索方向
        ↓
发现、筛选和解析资料
        ↓
抽取、清洗和归一化数据
        ↓
关联来源证据并进行质量检查
        ↓
查看结果、人工复核与导出
```

不同研究问题产生的字段和数据表并不相同。例如，太阳能电池任务可能包含“材料”“制备方法”“PCE”“稳定性”；超新星任务可能包含“光变曲线指标”“拟合结果”“前身星证据”。

## 工作台功能说明

创建任务后，可以在任务详情中使用以下页面：

| 页面 | 用途 |
| --- | --- |
| 总览 | 查看任务状态、来源数量、结构化记录、覆盖率和待复核事项。 |
| 来源 | 查看候选资料、筛选理由、下载和解析状态。 |
| 数据 | 按动态数据表浏览结构化记录，并按字段或来源筛选。 |
| 图像 / 图表 | 查看从 PDF 中识别的图像、图表数据与校验结果。 |
| 证据链 | 从记录回溯到来源文件、页码、章节或原文片段。 |
| 复核 | 处理证据不足、字段缺失或存在冲突的记录。 |
| 质量 | 查看覆盖率、警告、错误和字段缺口。 |
| 导出 | 下载可用的数据文件、调研报告和质量报告。 |
| 运行记录 | 查看任务执行过程中的阶段事件。 |

固定界面文案优先使用中文。论文题目、作者、来源名称、模型名称、原文证据和原始字段值会保留原始语言，以避免改变科研含义。

## 快速开始

### 1. 准备环境

建议使用 Python 3.11、Node.js 22 或更高版本。

```powershell
git clone <仓库地址> scidata-agent
cd scidata-agent

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\backend\requirements.txt

Copy-Item .\backend\.env.example .\backend\.env
```

在 `backend/.env` 中填写自己的模型服务凭证。至少需要设置：

```dotenv
DASHSCOPE_API_KEY=你的真实密钥
```

如使用特定工作空间、模型或服务地址，可在同一文件中调整对应配置。请勿提交 `.env`，也不要在截图、录屏或浏览器前端代码中暴露密钥。

### 2. 启动后端

打开一个终端：

```powershell
cd backend
python -m uvicorn scidata_agent.api.main:app --reload --port 8000
```

### 3. 启动前端

打开另一个终端：

```powershell
cd frontend
npm ci
npm run dev
```

在浏览器访问 `http://localhost:5173`。

### 4. 使用已保存的演示任务

如果仓库中已包含 `runtime/tasks/<任务ID>/` 与 `runtime/outputs/<任务ID>/`，启动服务后可在“历史任务”中直接打开它。无需重新执行模型调用，即可查看任务结果、证据、导出文件和运行记录。

迁移一个已完成任务时，请保留以下目录层级：

```text
runtime/
├─ tasks/
│  └─ <任务ID>/
│     ├─ task_state.json
│     └─ result_payload.json
└─ outputs/
   └─ <任务ID>/
      ├─ agent_monitor.jsonl
      ├─ agent_trace.json
      ├─ dynamic_records.csv
      └─ 其他任务结果文件
```

其中 `agent_monitor.jsonl` 用于“运行记录”页面；缺少该文件不会影响结果浏览，但运行时间线会为空。

## 创建和查看任务

### 完整分析

输入研究问题后，可选择上传 PDF、CSV、TSV、XLSX 或 XLS 文件。系统会结合上传内容和可用公开来源，完成检索、解析、抽取、核验和导出。

### 仅发现来源

如果暂时没有本地文件，可使用“仅发现来源”模式了解可用论文、数据库和检索方向。该模式生成候选来源和检索计划，适合开始研究前的资料摸排。

### 如何理解任务状态

| 状态 | 含义 |
| --- | --- |
| 排队中 / 运行中 | 任务正在等待或执行，可查看总览和运行记录。 |
| 已完成 | 工作流已结束，结果可浏览和导出。仍建议抽查关键记录的来源证据。 |
| 部分完成 | 已有可用结果，但仍存在字段缺口、证据不足或未完成的覆盖要求。 |
| 失败 / 已取消 | 任务未完整结束，可查看运行记录并按需要重新执行。 |

“来源已解析”只表示系统成功读取了资料，并不表示资料中的每个字段都已提取或已经过人工确认。

## 导出文件说明

“导出”页面会展示 CSV、JSON 和 Markdown 等数据文件。每个任务的内容不同，以下是常见文件：

| 文件 | 适用用途 |
| --- | --- |
| `dynamic_records.csv` | 全部清洗后的动态结构化记录汇总表。每行是一条记录，保留来源、页码、证据、置信度和本次任务的动态字段。 |
| `tables/*.csv` | 按数据表分别导出的 CSV。字段更集中，适合阅读某一类记录。 |
| `dynamic_records.json` | 动态记录的完整 JSON，适合程序处理或查看嵌套字段。 |
| `result.csv` | 仅包含符合数值指标结构的记录。某些以论文元数据、材料或方法为主的任务中，它可能为空。 |
| `result.json` | 任务结果、概要和相关状态信息。 |
| `source_catalog.json` | 候选来源、资料条目和处理状态。 |
| `evidence_traces.csv/json` | 记录与来源证据之间的关联关系。 |
| `quality_report.json`、`coverage_report.json` | 覆盖率、警告、冲突、字段缺口和质量检查结果。 |
| `needs_review.csv/json` | 需要人工确认的记录。 |
| `final_report.md` | 可直接阅读的调研报告。 |

`dynamic_records.csv` 是跨数据表的汇总文件，因此某些单元格为空是正常现象：一条“论文元数据”记录不会有“拟合参数”，一条“光变曲线拟合”记录也不会有作者字段。想聚焦某类数据时，请使用 `tables/` 下对应的 CSV。

## 本地部署

完成前端构建后，后端会同时提供网页和 API：

```powershell
cd frontend
npm run build

cd ..\backend
python -m uvicorn scidata_agent.api.main:app --host 0.0.0.0 --port 8000
```

然后访问 `http://localhost:8000`。

系统默认将任务、上传文件和导出结果保存在 `runtime/`。如需迁移或持久化保存，可设置 `SCIDATA_RUNTIME_DIR` 指向其他目录。

### Docker

```powershell
docker build -t scidata-agent .
docker run --rm -p 8000:8000 `
  -e DASHSCOPE_API_KEY="你的真实密钥" `
  -v "${PWD}\runtime:/data/runtime" `
  -e SCIDATA_RUNTIME_DIR=/data/runtime `
  scidata-agent
```

容器运行时请挂载持久化目录，否则容器重建后任务历史和导出结果将丢失。

## 使用提示

- 将“待复核”“警告”和“字段缺口”视为结果的一部分，而不是可以忽略的提示。
- 对用于论文、报告或决策的关键数值，请回到证据链中的原始来源核对。
- 模型服务、公开数据源的可用性和速率限制会影响任务耗时与结果覆盖度。
- 使用公开论文、数据库和第三方服务时，请遵守其访问许可、引用规范与使用条款。

## 开发者资料

- API 字段和接口说明：[backend/API_CONTRACT.md](backend/API_CONTRACT.md)
- 环境变量模板：[backend/.env.example](backend/.env.example)
- 后端测试：`cd backend; python run_tests.py`
- 前端测试与构建：`cd frontend; npm test; npm run build`
