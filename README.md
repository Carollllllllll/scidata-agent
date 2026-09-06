# SciData Agent

SciData Agent 是一个面向科研场景的多源数据检索、解析与结构化整合系统。用户输入研究问题，可选上传 PDF、CSV、TSV 或 Excel 文件；系统随后规划本次任务的数据结构，检索和筛选公开来源，解析文档、表格与图像，并输出带来源证据的结构化记录、质量审核结果和可下载交付物。

它的目标不是生成看似完整的答案，而是把每一条可用数据尽可能关联到论文、页码、章节、表格或图像等证据位置；证据不足、字段缺失和跨来源冲突会明确展示给用户复核。

## 核心能力

- **动态数据结构**：不预设固定学科模板。系统根据研究问题生成本次任务的字段组和数据表，例如“材料组成”“制备方法”“性能指标”“稳定性数据”。
- **多源资料发现**：可结合本地文件与联网检索，发现论文、开放数据库、数据集、补充材料、表格和图像等候选来源。
- **多模态解析**：支持 PDF 正文、PDF 表格、CSV/TSV/Excel，以及 PDF 中图表的定位、渲染、视觉模型读取和确定性校验。
- **证据与质量控制**：记录来源、页码、原文证据和抽取方式；分别统计字段覆盖率、证据覆盖率、冲突、警告、待复核项和停止条件。
- **可审计的智能体运行**：保存规划、决策、工具调用、停止原因和处理日志；任务可取消、重试或从检查点恢复。
- **面向用户的工作台**：提供中文优先的任务创建、进度、来源、动态数据表、图表、证据链、人工复核和导出界面。

## 工作流程

```text
研究问题 / 本地文件
        ↓
任务规划与动态 Schema 生成
        ↓
来源发现、筛选与资料获取
        ↓
PDF / 表格 / 数据文件 / 图表解析
        ↓
动态字段抽取、清洗、归一与去重
        ↓
来源追溯、跨模态核验、质量与覆盖审核
        ↓
人工复核队列、调研报告与全部导出文件
```

系统使用“观察 → 决策 → 工具执行”的运行方式推进任务。每个字段组都会先完成初始检索；若覆盖不足，系统可在受限次数内补充检索。任务不会因为页面需要展示而虚构记录或填充占位数据。

## 界面与使用方式

启动后打开工作台，可选择两种任务模式：

| 模式 | 适用情况 | 结果 |
| --- | --- | --- |
| 完整分析 | 有研究问题，且可选上传论文或数据文件 | 进行来源发现、解析、抽取、校验和导出 |
| 仅发现来源 | 暂无本地文件，只想先了解可用论文、数据库或检索方向 | 生成候选来源、检索计划和动态数据结构 |

完整分析模式支持上传 PDF、CSV、TSV、XLSX 和 XLS。默认安全限制为最多 20 个文件、单文件不超过 50 MiB、单次请求不超过 200 MiB；这些限制可通过环境变量调整。

任务详情页包括：

- **总览**：来源数量、结构化记录、证据覆盖、待复核项和动态数据结构。
- **来源**：候选来源的发现、选择、下载、解析状态，以及关联资料和选择理由。
- **数据**：按本次动态数据表浏览清洗记录或原始记录，支持筛选、搜索和打开证据详情。
- **图像 / 图表**：已识别图表、坐标轴、数据点、校验结果和跨模态互证。
- **证据链**：每条记录与来源、页码、章节、表格或图像之间的可追溯关系。
- **复核与质量**：人工复核队列、字段缺口、冲突、质量问题和覆盖审核。
- **导出与运行记录**：所有可公开下载的任务结果、Markdown 报告和最近运行事件。

### 中文化原则

固定产品文案、状态和常见字段名优先显示中文，例如“证据链追溯”“覆盖审核”“智能体运行状态”“制备方法”和“性能指标”。为避免改变科研含义，以下内容保留原始语言：论文题目、作者、来源名称、模型名称、字段原始值、错误详情和原文证据。

## 快速启动（本地开发）

### 1. 准备环境

建议使用 Python 3.11 和 Node.js 22 或更高版本。Windows PowerShell 示例：

```powershell
git clone <你的仓库地址> scidata-agent
cd scidata-agent

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\backend\requirements.txt

Copy-Item .\backend\.env.example .\backend\.env
```

在 `backend/.env` 中至少填写 DashScope/Qwen 配置：

```dotenv
DASHSCOPE_API_KEY=你的真实密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
QWEN_MODEL=qwen3.7-flash-2026-07-15
QWEN_VL_MODEL=qwen3-vl-235b-a22b-thinking
```

不要提交 `backend/.env`，也不要把 API Key 写进前端的 `VITE_*` 变量、截图或浏览器存储。完整的模型池、超时、并发、上传、缓存和运行上限配置见 [`backend/.env.example`](backend/.env.example)。

### 2. 启动后端

新开一个终端：

```powershell
cd D:\scidata-agent\backend
python -m uvicorn scidata_agent.api.main:app --reload --port 8000
```

后端默认使用 `runtime/` 保存任务状态、上传文件和导出结果。可设置 `SCIDATA_RUNTIME_DIR` 将其迁移到其他目录。

### 3. 启动前端

再开一个终端：

```powershell
cd D:\scidata-agent\frontend
npm ci
npm run dev
```

访问 `http://localhost:5173`。开发服务器会将 `/api` 请求代理到 `http://localhost:8000`。

### 4. 生产式本地运行

构建前端后，FastAPI 会从 `frontend/dist` 提供工作台页面：

```powershell
cd D:\scidata-agent\frontend
npm run build

cd ..\backend
python -m uvicorn scidata_agent.api.main:app --host 0.0.0.0 --port 8000
```

此时访问 `http://localhost:8000`。

## 任务状态与结果解读

| 状态 | 含义 | 建议 |
| --- | --- | --- |
| `queued` / `running` | 已排队或正在运行 | 查看当前阶段、运行记录和实时覆盖信息 |
| `completed` | 工作流完成，覆盖审核允许结束 | 仍应按科研要求抽查关键记录与来源证据 |
| `partial` | 已产生可检查的结果，但覆盖审核未满足停止条件 | 优先查看“质量”和“复核”中的字段缺口与建议动作 |
| `failed` / `cancelled` | 执行失败或已取消 | 查看运行记录；可重试或从检查点恢复 |

来源状态与记录质量是两套独立信息。某个来源“已解析”不等于其所有数据都可靠；某条记录“有证据”也不代表它已经完成全部字段覆盖或跨来源核验。

## 导出结果

每项任务的输出位于运行目录的任务专属目录中，前端“导出”页面会自动展示 CSV、JSON、Markdown 等结构化或可阅读结果文件。内部检查点和提取出的图片不会出现在下载列表中；图片应在“图像 / 图表”页面查看。

| 类别 | 典型文件 | 说明 |
| --- | --- | --- |
| 动态结构与记录 | `dynamic_schema.json`、`dynamic_records.json`、`dynamic_records.csv`、`dynamic_records_raw.json`、`tables/*.csv` | 本次任务动态 Schema、清洗记录、全部动态记录汇总 CSV、原始记录及每个动态表的 CSV |
| 指标型结果 | `result.csv`、`result.json` | `result.csv` 只包含符合严格数值指标条件的记录；如果任务以论文元数据、材料或方法等字段为主，它可能为空，应使用 `tables/*.csv` 或 `dynamic_records.json` |
| 来源与调研 | `source_catalog.json`、`source_discovery_plan.json`、`source_selection_plan.json`、`source_triage.json`、`paper_survey.csv/json` | 候选来源、筛选/下载决策和按论文聚合的调研结果 |
| 图表与证据 | `figures/`、`chart_extractions.json`、`chart_data/`、`chart_validation_report.json`、`cross_modal_validation.json` | 图像、图表抽取数据、图表校验与文本/表格/图像互证 |
| 质量与人工复核 | `quality_report.json`、`coverage_report.json`、`needs_review.json/csv` | 覆盖率、冲突、警告、字段缺口和待人工确认项目 |
| 运行审计与报告 | `agent_trace.json`、`decision_history.json`、`tool_history.json`、`processing_log.json`、`final_report.md` | 智能体决策和工具调用轨迹、处理日志及可阅读的最终报告 |

不要只以 CSV 是否为空判断任务是否成功。首先查看“数据”页的动态表、`dynamic_records.json` 以及质量报告；不同研究问题生成的数据结构并不相同。

## API 概览

接口基址默认为 `http://localhost:8000`。完整字段、错误码和安全约束见 [`backend/API_CONTRACT.md`](backend/API_CONTRACT.md)。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/health` | 服务、模型配置和 Agent 阶段健康检查；不返回密钥 |
| `POST` | `/api/analyze` | 创建完整分析任务（`multipart/form-data`） |
| `POST` | `/api/discover` | 创建仅发现来源任务 |
| `GET` | `/api/tasks` | 查询任务历史 |
| `GET` | `/api/tasks/{task_id}` | 查询任务状态、结果、质量和下载 URL |
| `GET` | `/api/tasks/{task_id}/events` | 查询最近运行事件；可用 `tail` 控制数量 |
| `POST` | `/api/tasks/{task_id}/cancel` | 协作式取消排队或运行中的任务 |
| `POST` | `/api/tasks/{task_id}/retry` | 基于安全的已上传输入创建重试任务 |
| `POST` | `/api/tasks/{task_id}/resume` | 从同一任务的最近有效检查点恢复 |
| `POST` | `/api/tasks/{task_id}/reviews/{review_id}` | 写入人工复核结论和备注 |
| `GET` | `/api/tasks/{task_id}/export?format=...` | 下载标准导出文件 |
| `GET` | `/api/tasks/{task_id}/assets/{scope}/{asset_path}` | 获取任务内的公开上传或输出资源 |

`POST /api/analyze` 支持研究问题、重复 `files` 字段，以及 PDF 页数、资源数量、文本块、图表数量、并发、是否联网检索、是否自动下载、是否复用动态记录等可选参数。除服务安全边界外，科学数据处理上限均需显式设置；`0` 或省略通常表示不主动截断。

## 命令行运行

以下命令在 `backend/` 目录执行：

```powershell
# 只生成来源发现计划
python -m scidata_agent.cli `
  --question "比较钙钛矿太阳能电池的材料、制备方法、PCE 与稳定性" `
  --discover-only `
  --output-dir ..\outputs

# 解析本地文件并进行完整分析
python -m scidata_agent.cli `
  --question "从上传论文和表格中提取材料、方法、性能指标与来源证据" `
  --files ..\examples\demo_scientific_paper.pdf ..\examples\perovskite_metrics.csv `
  --output-dir ..\outputs
```

`--allow-rule-fallback` 仅用于本地测试；正式结果应配置 Qwen，避免把规则回退结果当作模型驱动的科研结论。

## Docker 与 SAE 部署

根目录的 [`Dockerfile`](Dockerfile) 使用多阶段构建：先构建 React 前端，再由 Python/FastAPI 在同一服务中提供静态页面和 API。

```powershell
cd D:\scidata-agent
docker build -t scidata-agent .
docker run --rm -p 8000:8000 `
  -e DASHSCOPE_API_KEY="你的真实密钥" `
  -v "${PWD}\runtime:/data/runtime" `
  scidata-agent
```

容器环境请配置 `SCIDATA_RUNTIME_DIR=/data/runtime` 并挂载持久化存储；否则容器重建或实例替换后任务历史、上传文件和导出结果会丢失。

若使用阿里云 SAE 的 Python 代码包部署，先构建前端，再生成 ZIP：

```powershell
cd D:\scidata-agent\frontend
npm ci
npm run build

cd ..
.\scripts\build_sae_code_package.ps1
```

生成的 ZIP 根目录包含 `requirements.txt`、`backend/scidata_agent` 和 `frontend/dist`。SAE 可选择 Python 3.11，并使用以下启动命令：

```sh
cd backend && python -m uvicorn scidata_agent.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

在 SAE 环境变量中配置密钥和运行目录，不要把 `.env` 打进代码包或镜像。公开部署建议设置 `SCIDATA_API_TOKEN` 并在可信反向代理后谨慎配置 CORS 与转发头。

## 测试与构建

```powershell
# 后端测试
cd D:\scidata-agent\backend
python run_tests.py

# 前端单元测试和生产构建
cd D:\scidata-agent\frontend
npm test
npm run build
```

后端测试覆盖动态 Schema、来源发现、PDF/表格解析、图表抽取与校验、质量报告、来源去重、并发处理、结构化输出和运行日志。前端测试覆盖任务详情、导出、质量和证据相关组件。

## 目录结构

```text
backend/
  scidata_agent/       # Agent、连接器、解析器、质量控制与 FastAPI API
  API_CONTRACT.md      # API 详细契约
  .env.example         # 配置项示例
  tests/               # 后端测试
frontend/
  src/                 # React 工作台
  public/              # 静态资源
scripts/
  build_sae_code_package.ps1
runtime/               # 运行时任务数据（默认不提交）
Dockerfile
```

## 使用边界

- 系统输出是科研数据整理和证据追溯辅助结果，不替代领域专家审阅或同行评议。
- `partial`、警告、冲突和待复核项应被视为结果的一部分，而不是可以忽略的页面提示。
- 模型与外部数据源的可用性、速率限制和内容质量会影响运行时间和最终覆盖率。
- 请遵守论文、数据库和第三方服务的访问许可与引用要求。
