# SciData Agent 当前待做清单

> 本文件以当前项目代码、SciData Agent 任务书和 API 适配层为基准。
>
> 核心闭环：
>
> ~~~text
> 用户问题 -> 多源发现 -> 来源筛选 -> 资料解析
> -> 动态字段抽取 -> Schema 对齐 -> 证据绑定
> -> 质量校验 -> CSV/JSON 导出
> ~~~

## 1. 当前状态

### 已具备

- Qwen 任务规划和动态字段 Schema。
- arXiv、OpenAlex、Semantic Scholar、Crossref、Figshare、GitHub、Zenodo 等来源连接器。
- 多源搜索、来源选择、Triage 和 Source Catalog。
- PDF 文本、章节、CSV、TSV、Excel 解析。
- TATR PDF 表格解析和 fallback 路径。
- Qwen-VL 图表识别和图表数据抽取。
- 动态记录、字段归一、证据绑定、冲突检测和质量报告。
- CSV、JSON、Markdown、质量报告和处理日志导出。
- Qwen 文本/VL 模型池切换。
- FastAPI 后台任务、任务状态、CORS、上传隔离和下载白名单。

### 当前缺口

- 前端工作台尚未完成。
- 真实 API 端到端测试尚未完整完成。
- 部分来源仍然停留在元数据、README 或文件清单层面。
- 补充材料、开放数据库、网页附件和独立图片的深入解析不完整。
- 图表二次复核、OCR、断点续跑和人工反馈尚未完成。
- 知识图谱不是赛题硬性要求，本阶段不作为主线。

---

# 2. 前端待做清单

## 2.1 总体目标

前端应是科研数据整合工作台，而不是普通聊天页面或单纯 CSV 查看器。用户应能：

1. 输入科研问题并上传资料。
2. 启动调研任务。
3. 看到 Agent 当前执行节点和进度。
4. 查看找到的来源及其选择理由。
5. 查看正文、表格、图表和附件中的结构化结果。
6. 点击字段查看原文证据。
7. 识别 warning、conflict 和 needs_review。
8. 下载 CSV、JSON、质量报告和最终报告。

## 2.2 设计原则

### 任务中心

首页直接进入工作台，不做营销式落地页。主要入口是新建任务、历史任务、运行状态和结果。

自然语言输入框可以存在，但聊天回答不是主结果。结构化记录、来源和证据必须优先展示。

### 证据中心

每个关键字段尽量展示来源名称、原始链接、文件名、页码或表格行号、原文证据、抽取置信度、warning 和校验状态。

不能只展示模型结论而隐藏证据。

### 动态字段

前端不能固定为材料科学、Try-on、天文或某一种任务的字段。必须读取 dynamic_extraction_plan、dynamic_tables、dynamic_records 以及字段名称、类型、描述、必填状态和证据要求。

### 状态分离

来源状态：

~~~text
discovered / selected / downloaded / parsed / failed / skipped
~~~

数据质量状态：

~~~text
verified / needs_review / warning / conflict / error
~~~

来源已解析不代表其中所有记录可信；两类状态必须分开显示。

### 失败透明

下载失败、模型失败、字段未找到、TATR 未加载、图表无法识别时，前端必须显示真实原因。不能把只获得元数据显示成已完成深度解析，不能用假数据填充空结果。

### 科研数据工作台风格

- 结果表支持比较、筛选、排序和分页。
- 重要字段固定在左侧。
- 长证据放在抽屉或详情面板。
- 支持动态列，不强制固定字段。
- 长标题和长证据可折叠。
- 不使用多层卡片堆叠内容。
- Loading 需要显示当前节点和数量。
- 中文、英文、单位、公式和长文本都要能正常显示。

---

## 2.3 页面与交互

### 页面一：科研工作台首页

内容：

- 新建任务。
- 最近任务。
- 任务状态筛选。
- 后端健康状态。
- 非敏感的模型配置状态。

交互：

- 新建任务进入输入页。
- 历史任务进入任务详情。
- 后端不可用显示重试按钮。
- 绝不显示 API Key。

API：

~~~text
GET /api/health
~~~

### 页面二：新建调研任务

输入：

- research_question：必填的自然语言科研问题。
- files[]：可选的 PDF、CSV、TSV、XLSX。
- max_pdf_pages。
- max_arxiv_papers。
- max_dynamic_text_blocks。
- max_record_text_blocks。
- max_figures_per_pdf。

交互流程：

1. 检查问题不能为空。
2. 选择或拖拽文件。
3. 显示文件名、类型、大小和删除按钮。
4. 设置可选参数。
5. 点击开始分析。
6. 禁用重复提交。
7. 获得 task_id 后跳转任务进度页。

API：

~~~text
POST /api/analyze
Content-Type: multipart/form-data
~~~

立即返回：

~~~json
{
  "task_id": "20260819_120000_123_abcd",
  "status": "queued",
  "status_url": "/api/tasks/20260819_120000_123_abcd",
  "events_url": "/api/tasks/20260819_120000_123_abcd/events"
}
~~~

异常处理：

- 网络错误：保留表单并允许重试。
- 不支持的文件：提交前提示。
- 文件过大：显示明确原因。
- 服务端失败：显示后端错误，不清空用户输入。
- 刷新页面不能重新提交同一任务。

### 页面三：任务进度页

API：

~~~text
GET /api/tasks/{task_id}
GET /api/tasks/{task_id}/events?tail=100
~~~

轮询：

- queued 或 running 时每 1-3 秒轮询。
- completed 或 failed 时停止。
- 页面刷新后根据 task_id 恢复。
- 后台标签页可以降低轮询频率。

状态：

~~~text
queued      已排队
running     运行中
completed   已完成
failed      失败
~~~

阶段：

~~~text
ensure_llm_ready
task_planning
dynamic_schema_planning
source_discovery
multi_source_search_planning
multi_source_search
source_selection
source_triage
multi_source_ingestion
arxiv_pdf_ingestion
artifact_action_planning
artifact_action_execution
source_parsing
section_interpretation
figure_chart_extraction
dynamic_extraction
record_extraction
normalization
provenance_tracking
quality_validation
export
~~~

显示 status、current_step、message、progress、summary 和最近 events。

### 页面四：来源发现页

数据：

- source_catalog。
- discovered_sources。
- source_selection_plan。
- source_triage_decisions。
- connector_status。

来源字段：

- 标题、来源类型、Provider、原始链接。
- 相关性分数和 LLM 选择理由。
- 匹配的用户需求。
- 当前状态、推荐动作。
- 下载和解析状态。
- 失败原因。

筛选：

- 来源类型。
- Provider。
- 相关性。
- 处理状态。
- 失败来源。
- 已选来源。
- 需要继续解析的来源。

API：

~~~text
GET /api/tasks/{task_id}
GET /api/tasks/{task_id}/events
~~~

来源详情抽屉需要展示标题、链接、类型、Provider、相关性理由、元数据、Artifact、下载状态、解析状态、失败信息和关联记录。

### 页面五：结果总览

数据：

- summary。
- records。
- dynamic_records。
- source_catalog。
- quality_report。
- download_urls。

统计：

~~~text
处理文件数 / 发现来源数 / 成功解析来源数
文本块数 / 表格数 / 图表数 / 抽取记录数
动态表数量 / warning 数 / conflict 数
needs_review 数 / 证据覆盖率
~~~

结果表需要支持动态列、列显示隐藏、字段筛选、来源筛选、置信度排序、warning/conflict 筛选、分页或虚拟滚动以及点击记录打开证据面板。

不要假设固定存在 material、method 或 metric，不要把空值自动补成事实。

### 页面六：动态数据表

数据：

- dynamic_extraction_plan.dynamic_tables。
- dynamic_records。
- dynamic_records_clean。
- tables/ 导出结果。

左侧显示表名、说明、实体类型、优先级、字段数量和记录数量。

右侧显示动态列，并提供字段说明、类型、必填状态、证据要求和缺失数量。

交互：

- 切换动态表。
- 调整列宽。
- 隐藏和显示字段。
- 导出当前表。
- 点击记录查看证据。
- 对比原始记录和清洗后记录。

### 页面七：证据详情

必须显示：

- 记录 ID、实体、字段名和值。
- 单位和实验条件。
- 来源文件和来源类型。
- 页码、表格行号或图表编号。
- 原文证据。
- 置信度、warning、归一化状态和冲突状态。

证据类型：

~~~text
PDF 正文 / PDF 章节 / PDF 表格
CSV/Excel 行 / 图表数据点
来源元数据 / GitHub README / 补充材料
~~~

交互：

- 结果表点击记录打开。
- 来源页点击来源反向查看记录。
- 证据可复制、展开和折叠。
- 有 PDF 预览时支持跳页；没有时至少显示页码。
- 缺失证据必须显示警告，不能隐藏。

### 页面八：图表和表格解析

图表读取：

~~~text
figures/
chart_extractions.json
chart_validation_report.json
chart_data/
~~~

展示图像、Caption、类型、坐标轴、单位、图例、数据点数量、VL 置信度、验证结果和 needs_review。

表格展示原始表格或截图、结构化表格、页码、TATR 加载状态、识别方法、是否 fallback、行列数量、表头和错位警告。

框架图、流程图、照片和定性图显示为非数值图表，不算识别失败。无法确认的数字不能显示成确定值。

### 页面九：质量报告

数据：

- quality_report。
- needs_review。
- conflicts。
- chart_validation_report。
- connector_status。

统计：

~~~text
记录总数 / 问题数 / warning 数 / error 数 / conflict 数
证据覆盖率 / 值证据覆盖率 / 字段覆盖率
来源数 / 失败来源数
~~~

问题类型：

~~~text
证据缺失 / 字段缺失 / 低置信度
单位缺失 / 单位异常 / 值证据不一致
跨来源冲突 / 图表坐标轴异常
表格结构异常 / 来源下载失败 / 模型调用失败
~~~

交互：

- 点击问题定位记录。
- 点击冲突查看不同来源的值。
- 点击来源失败查看连接器错误。
- 按 warning、error、conflict 筛选。
- 加入复核队列。

### 页面十：导出和调研报告

支持：

~~~text
csv / json / quality_report / processing_log
source_catalog / paper_survey / dynamic_schema
dynamic_records / needs_review / chart_extractions
chart_validation / summary / final_report
~~~

显示文件名、类型、说明、生成状态、下载按钮和失败原因，并提供 final_report.md 预览。

前端必须使用 download_urls，不能使用后端 Windows 绝对路径。

---

## 2.4 API 交互总表

| 场景 | API | 方法 | 主要数据 | 说明 |
|---|---|---|---|---|
| 检查服务 | /api/health | GET | status、模型状态、版本 | 首页使用 |
| 创建完整调研 | /api/analyze | POST multipart | research_question、files、参数 | 返回 202 和 task_id |
| 创建来源发现 | /api/discover | POST form | research_question | 来源发现模式 |
| 查询任务 | /api/tasks/{task_id} | GET | status、current_step、summary | 轮询核心 |
| 查看事件 | /api/tasks/{task_id}/events | GET | events[] | 详细进度 |
| 下载 CSV | /api/tasks/{task_id}/export?format=csv | GET | 文件流 | 使用 download_urls.csv |
| 下载 JSON | /api/tasks/{task_id}/export?format=json | GET | 文件流 | 使用 download_urls.json |
| 下载质量报告 | /api/tasks/{task_id}/export?format=quality_report | GET | 文件流 | 使用 download_urls.quality_report |
| 下载报告 | /api/tasks/{task_id}/export?format=final_report | GET | 文件流 | 使用 download_urls.final_report |

## 2.5 前端状态机

~~~text
idle -> submitting -> queued -> running -> completed -> viewing_result

submitting -> submit_failed
queued/running -> failed
completed -> export_failed
completed -> needs_review
~~~

状态要求：

- submitting：禁用重复提交并显示上传状态。
- queued：显示排队。
- running：显示 Agent 节点和数量。
- completed：停止轮询并加载结果。
- failed：显示失败节点、错误和日志入口。
- needs_review：任务成功但部分记录待复核。
- export_failed：只标记下载失败，不把分析任务标记为失败。

## 2.6 前后端职责边界

### 前端负责

- 用户输入和交互。
- 页面状态管理。
- API 请求和轮询。
- 来源、结果、证据和质量可视化。
- 下载入口。
- 真实展示失败和警告。

### 前端不负责

- 调用 Qwen API。
- 调用 TATR 或 VL。
- 直接解析 PDF。
- 读取服务器本地路径。
- 自己判断科研字段。
- 用假数据填补缺失结果。
- 重新实现来源相关性排序。

---

# 3. 后端待做清单

## 3.1 P0：前端联调前

- [ ] 安装 pytest，运行适配层测试和现有测试。
- [ ] 增加 /api/analyze 的真实 FastAPI 接口测试。
- [ ] 测试 queued -> running -> completed。
- [ ] 测试 failed 状态和错误信息。
- [ ] 测试同名上传文件不会覆盖。
- [ ] 测试非法 task_id 和非法导出格式不能越过白名单。
- [ ] 测试 CORS 可被前端开发服务器使用。
- [ ] 测试所有 download_urls 能下载。
- [ ] 确认 CLI 旧调用方式不受影响。
- [ ] 增加真实 PDF + CSV/Excel 端到端案例。
- [ ] 保存一套固定输出供前端联调。
- [ ] 统一参数错误、任务失败、来源失败和导出失败的错误结构。
- [ ] 明确 /api/discover 是计划模式还是完整搜索模式。

## 3.2 P0：赛题核心能力稳定性

- [ ] 验证真实 Qwen 参与规划和动态抽取。
- [ ] 验证真实 Qwen-VL 参与图表识别。
- [ ] 验证真实 TATR 加载和推理，并记录是否 fallback。
- [ ] 识别 HTTP 200 但 JSON 内部返回的配额耗尽错误。
- [ ] 完善重试、模型切换和最终失败记录。
- [ ] 记录连接器请求、返回数量、失败原因和重试次数。
- [ ] 明确只获得元数据、未读取正文的 warning。
- [ ] 关键字段缺证据时降置信度或进入复核。
- [ ] 确认数值必须能在证据中找到，或被标记为待复核。
- [ ] 完善跨来源冲突检测。
- [ ] 完善字段、单位和实体名称归一。
- [ ] 缺失字段使用 null 或 warning，不编造数据。

## 3.3 P1：多源深度解析

- [ ] 补充材料下载和解析。
- [ ] 独立表格附件统一解析。
- [ ] 开放数据库 API 或结构化返回读取。
- [ ] 通用网页正文解析。
- [ ] GitHub 关键文件读取，不只读取 README 和 manifest。
- [ ] 独立图像和图表附件解析。
- [ ] 分别记录来源的发现、下载、读取和解析状态。
- [ ] 单个来源失败时继续其他来源。
- [ ] 完善资源缓存，避免重复下载。
- [ ] 完善多 PDF 的章节、表格和图表隔离，避免串台。

## 3.4 P1：表格和图表质量

- [ ] 保存 TATR 检测和结构模型的加载状态。
- [ ] 记录每张表的 extraction_method。
- [ ] 对比 TATR 和 fallback 结果质量。
- [ ] 改善表头、合并单元格和多行表头处理。
- [ ] 检测表格行列错位并生成 warning。
- [ ] 区分定量图表、框架图、流程图、照片和定性图。
- [ ] 验证图表坐标轴、单位、图例和数据点。
- [ ] 对异常图表进行 VL 二次复核。
- [ ] 保存初次抽取、复核抽取和最终结果。
- [ ] 增加扫描 PDF 和图片表格的 OCR 路径。

## 3.5 P1：任务可靠性

- [ ] 将任务状态持久化到稳定存储。
- [ ] 增加任务取消。
- [ ] 增加单节点重试。
- [ ] 增加任务级断点续跑。
- [ ] 为文本块、表格、图表和记录抽取保存检查点。
- [ ] 支持从失败节点继续。
- [ ] 增加并发任务数量控制。
- [ ] 增加单任务资源限制。
- [ ] 增加大文件和旧任务清理策略。

## 3.6 P1：结果和质量

- [ ] 明确原始记录、清洗记录和最终记录的区别。
- [ ] 完善 needs_review.csv/json。
- [ ] 增加字段、证据和来源覆盖率。
- [ ] 增加来源级质量评分。
- [ ] 增加重复合并的可解释日志。
- [ ] 增加单位转换和单位冲突报告。
- [ ] 增加没有找到可用数据的明确状态。
- [ ] 对质量检查进行真实端到端验证。

## 3.7 P2：可选增强

- [ ] 知识图谱 nodes.json 和 edges.json。
- [ ] 人工确认、修改和反馈 API。
- [ ] 反馈后重新校验或重新抽取。
- [ ] 更完整网页和附件发现。
- [ ] OCR 和复杂版面增强。
- [ ] 向量检索或语义缓存。
- [ ] SQLite/PostgreSQL 持久化。
- [ ] 任务历史搜索和用户工作区。
- [ ] 用户权限和鉴权。

## 3.8 参赛材料

- [ ] PDF 单源测试案例。
- [ ] CSV/Excel 测试案例。
- [ ] PDF + 表格多源案例。
- [ ] 自动检索真实来源案例。
- [ ] 冲突和缺失字段案例。
- [ ] 保存每个案例的输入、日志、结果和质量报告。
- [ ] 系统架构图和 Agent 工作流图。
- [ ] 来源和证据链示意图。
- [ ] TATR 和 Qwen-VL 真实运行证明。
- [ ] API 使用说明。
- [ ] 技术报告、答辩 PPT 和端到端演示视频。

---

# 4. 两名前端开发者的协作边界

## 开发者 A：任务与来源工作台

负责目录：

~~~text
src/features/task/
src/features/progress/
src/features/discovery/
src/features/sources/
~~~

负责页面：

~~~text
工作台首页
新建任务页
任务列表页
任务进度页
来源发现页
来源详情页
~~~

主要接口：

~~~text
GET  /api/health
POST /api/analyze
POST /api/discover
GET  /api/tasks/{task_id}
GET  /api/tasks/{task_id}/events
~~~

## 开发者 B：结果与证据工作台

负责目录：

~~~text
src/features/results/
src/features/tables/
src/features/evidence/
src/features/charts/
src/features/quality/
src/features/exports/
~~~

负责页面：

~~~text
结果总览页
动态数据表页
证据详情页
图表和表格页
质量报告页
导出和调研报告页
~~~

主要接口：

~~~text
GET /api/tasks/{task_id}
GET /api/tasks/{task_id}/export?format=csv
GET /api/tasks/{task_id}/export?format=json
GET /api/tasks/{task_id}/export?format=quality_report
GET /api/tasks/{task_id}/export?format=final_report
GET /api/tasks/{task_id}/export?format=chart_extractions
~~~

## 共同维护

- [ ] src/api/client.ts：统一 API 请求封装。
- [ ] src/types/api.ts：统一 TypeScript 类型。
- [ ] 状态徽章、动态表格、证据面板、空状态和错误状态组件。
- [ ] 颜色、字体、间距、按钮、表格和警告样式。
- [ ] 路由参数命名和任务详情布局。
- [ ] 统一 mock 数据格式，避免两个人构造不同响应。

---

# 5. 前端联调验收清单

- [ ] 可以提交只有研究问题、没有本地文件的任务。
- [ ] 可以提交多个 PDF/CSV/Excel 文件。
- [ ] 同名文件不会导致结果混淆。
- [ ] 提交后立即显示 queued。
- [ ] 可以看到 running 和当前 Agent 节点。
- [ ] 刷新浏览器后可以恢复 task_id 对应任务。
- [ ] 失败时看到失败节点和真实错误。
- [ ] 完成后看到来源、记录和质量统计。
- [ ] 来源列表和最终记录使用同一个 task_id。
- [ ] 动态字段能正确生成表格列。
- [ ] 点击记录能打开证据面板。
- [ ] 点击来源能看到关联记录。
- [ ] 能区分真实解析和 fallback。
- [ ] warning、conflict、needs_review 不被隐藏。
- [ ] CSV、JSON、质量报告和最终报告可以下载。
- [ ] 不依赖后端 Windows 绝对路径。
- [ ] 空结果显示未找到可用数据，不显示假数据。
- [ ] 中文和英文问题都能正常显示。
- [ ] 长标题、长证据和动态列不会重叠或溢出。

# 6. 最终交付判断

达到可参赛演示状态至少需要：

~~~text
前端任务与来源工作台
前端结果与证据工作台
API 联调测试
三个以上真实端到端案例
稳定的来源失败和模型失败提示
CSV/JSON/质量报告导出
技术报告、PPT 和演示视频
~~~

知识图谱、人工反馈、OCR 和复杂网页解析属于增强方向，不应阻塞上述主链路。
