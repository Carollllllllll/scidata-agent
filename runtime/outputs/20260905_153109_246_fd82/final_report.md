# SciData Agent Final Report

## Task

- Task ID: `20260905_153109_246_fd82`
- Research question: 研究 Ia 型超新星光变曲线，整合峰值星等、衰减率和来源证据
- Files processed: 139
- Text blocks processed: 519
- Heading candidates extracted: 1149
- Section blocks processed: 682
- Discovered sources: 1284
- Multi-source search requests: 14
- LLM source selection decisions: 83
- LLM-selected sources before safety triage: 35
- Source triage decisions: 1284
- Sources selected for ingestion: 20
- Source insights: 0
- Connector checks: 112 / failed: 31
- Downloaded PDFs: 49
- Parseable downloaded files: 49
- Metric records extracted: 11
- Dynamic records after cleaning: 228
- Dynamic records raw: 182
- Needs review records: 150
- Dynamic tables with data: 6

## Run Alerts

- Multi-source search was partial: 31 connector request(s) failed (figshare, semantic_scholar, figshare, semantic_scholar, figshare, figshare, semantic_scholar, semantic_scholar).
- Figures were detected in PDFs but no chart data was extracted. Check Qwen-VL configuration (QWEN_VL_MODEL) or classifier decisions in chart_extractions.json.

## Dynamic Extraction Schema

- Domain: astronomy
- Task type: data_extraction
- User focus: peak_magnitude, decay_rate, progenitor_model, host_galaxy_type, observation_telescope, filter_band, redshift

| Table | Entity | Fields |
|---|---|---|
| light_curve_metrics | metric | supernova_name, redshift, peak_magnitude, filter_band, decay_rate, observation_telescope, host_galaxy_type |
| progenitor_evidence | evidence | supernova_name, progenitor_model |
| light_curve_fits | method | supernova_name, fit_method, fit_parameters, residual_std_dev, chi_squared |
| cross_survey_comparisons | limitation | supernova_name, metric_name, source_1, value_1, source_2, value_2, discrepancy_reason |
| other_required_fields | other | paper_title, material, method, metric_value, unit, condition |

## Connector Status

- Checked: 112
- Failed: 31

| Connector | Status | Added | Query | Error |
|---|---|---|---|---|
| arxiv | completed | 100 | Ia supernova light curve peak magnitude decline rate |  |
| openalex | completed | 100 | Ia supernova light curve AND (peak magnitude OR decay rate) AND (host galaxy OR progenitor) |  |
| zenodo | completed | 24 | Ia supernova light curve data set |  |
| figshare | failed | 0 | Ia supernova supplementary data | JSON request failed: url=https://api.figshare.com/v2/articles/search, attempts=3, error=HTTP Error 403: Forbidden |
| github | completed | 0 | Ia supernova light curve analysis repository |  |
| semantic_scholar | failed | 0 | Ia supernova origin progenitor model host galaxy metallicity | JSON request failed: url=https://api.semanticscholar.org/graph/v1/paper/search?query=Ia+supernova+origin+progenitor+model+host+galaxy+metallicity&limit=100&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2Curl%2CopenAccessPdf%2CexternalIds... |
| crossref | completed | 83 | Ia supernova light curve peak magnitude decay rate |  |
| arxiv | completed | 100 | Type Ia supernova light curve peak magnitude decline rate |  |
| openalex | completed | 99 | Type Ia supernova light curve peak magnitude decay rate |  |
| semantic_scholar | completed | 90 | Ia supernova light curve shape and luminosity relation |  |
| zenodo | completed | 24 | Type Ia supernova light curve data multi-band photometry |  |
| figshare | failed | 0 | Ia supernova light curve tables supplementary materials | JSON request failed: url=https://api.figshare.com/v2/articles/search, attempts=3, error=HTTP Error 403: Forbidden |
| github | completed | 0 | Ia supernova light curve analysis code repository |  |
| arxiv | completed | 0 | Type Ia supernova light curve peak magnitude decline rate |  |
| openalex | completed | 56 | Type Ia supernova light curve shape and luminosity correlation |  |
| semantic_scholar | failed | 0 | Ia supernova progenitor models and ejecta composition | JSON request failed: url=https://api.semanticscholar.org/graph/v1/paper/search?query=Ia+supernova+progenitor+models+and+ejecta+composition&limit=100&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2Curl%2CopenAccessPdf%2CexternalIds%2Ccita... |
| zenodo | completed | 7 | Type Ia supernova photometry light curves Open Supernova Catalog |  |
| figshare | failed | 0 | Type Ia supernova light curve parameters table | JSON request failed: url=https://api.figshare.com/v2/articles/search, attempts=3, error=HTTP Error 403: Forbidden |
| arxiv | completed | 69 | Ia supernova spectroscopic diversity peak luminosity correlation |  |
| zenodo | completed | 17 | ZTF SN Ia DR2 photometry light curves |  |

## Multi-source Discovery

- Search requests: 14
- Candidate sources: 1284
- Providers: arxiv(449), crossref(150), github(1), open_database(5), openalex(506), paper_search(1), repository(1), semantic_scholar(78), supplementary_material(1), zenodo(92)

| Provider | Type | Title | URL | Query |
|---|---|---|---|---|
|  | open_database | The Supernova Legacy Survey (SNLS) | https://snls.physics.ubc.ca/ |  |
|  | open_database | Las Cumbres Observatory Global Telescope Network (LCOGT) | https://lco.global/ |  |
|  | open_database | Open Supernova Catalog | https://www.open supernova catalog.org/ |  |
|  | open_database | NASA Exoplanet Archive - Supernovae Section | https://exoplanetarchive.ipac.caltech.edu/ |  |
|  | paper_search | arXiv:astro-ph.HE - 超新星相关论文 | https://arxiv.org/list/astro-ph.HE/recent | Ia supernova light curve peak magnitude decline rate |
|  | supplementary_material | Supplementary Materials of ApJ Papers on SN Ia | https://iopscience.iop.org/article/10.3847/1538-4357/abf6d2/suprep |  |
|  | open_database | SIMBAD Astronomical Database | http://simbad.u-strasbg.fr/simbad/ | Ia supernova |
|  | repository | GitHub - Ia Supernova Light Curve Repository | https://github.com/astrolab/ia-sn-lightcurves |  |
| arxiv | paper | Type Ia Supernova Models and Progenitor Scenarios | http://arxiv.org/abs/1302.3371v2 | Ia supernova light curve peak magnitude decline rate |
| arxiv | paper | Timescale Stretch Parameterization of Type Ia Supernova B-band Light Curves | http://arxiv.org/abs/astro-ph/0104382v1 | Ia supernova light curve peak magnitude decline rate |
| arxiv | paper | The Standardizability of Type Ia Supernovae in the Near-Infrared: Evidence for a Peak Luminosity-Decline Rate Relation in the Near-Infrared | http://arxiv.org/abs/1201.2913v2 | Ia supernova light curve peak magnitude decline rate |
| arxiv | paper | CfA3: 185 Type Ia Supernova Light Curves from the CfA | http://arxiv.org/abs/0901.4787v5 | Ia supernova light curve peak magnitude decline rate |
| arxiv | paper | Pre-nebular light curves of type I supernovae | http://arxiv.org/abs/1611.08746v2 | Ia supernova light curve peak magnitude decline rate |
| arxiv | paper | Testing Cosmic Acceleration with Type Ia Supernovae | http://arxiv.org/abs/astro-ph/0101521v1 | Ia supernova light curve peak magnitude decline rate |
| arxiv | paper | Time Dilation in the Light Curve of the Distant Type Ia Supernovae SN 1995K | http://arxiv.org/abs/astro-ph/9605134v1 | Ia supernova light curve peak magnitude decline rate |
| arxiv | paper | Probing type Ia supernova properties using bolometric light curves from the Carnegie Supernova Project and the CfA Supernova Group | http://arxiv.org/abs/1811.08969v1 | Ia supernova light curve peak magnitude decline rate |
| arxiv | paper | Topological Control of Chirality and Spin with Structured Light | http://arxiv.org/abs/2508.08733v3 | Ia supernova light curve peak magnitude decline rate |
| arxiv | paper | The light curves of type Ia Supernova 2004fu | http://arxiv.org/abs/astro-ph/0606051v2 | Ia supernova light curve peak magnitude decline rate |
| arxiv | paper | Type Ia Supernova Progenitors, Environmental Effects, and Cosmic Supernova Rates | http://arxiv.org/abs/astro-ph/9907386v1 | Ia supernova light curve peak magnitude decline rate |
| arxiv | paper | UBVRI Light Curves of 44 Type Ia Supernovae | http://arxiv.org/abs/astro-ph/0509234v1 | Ia supernova light curve peak magnitude decline rate |

## Source Selection

- Decisions: 83
- Decision counts: deep_read(34), metadata_only(42), read_readme(1), reject(6)
- Priority counts: high(28), low(23), medium(32)
- Time range interpreted: not specified
- Summary: 优先选择高置信度、包含多波段光度数据、宿主星系信息及前身星模型推断的综合性研究。重点筛选Open Supernova Catalog、ZTF SN Ia DR2、CfA3等权威数据库与最新论文（如2025-2026年），确保覆盖峰值星等、衰减率、红移、滤波器、观测望远镜、宿主星系类型及前身星模型等关键字段。拒绝因权限问题无法访问的Figshare资源，对DOI已验证的论文优先深度阅读。 优先选择包含多波段光度数据、宿主星系信息和前身星模型推断的综合性研究，特别是近五年内发表且被引量高的论文。重点筛选能提供峰值星等、衰减率与前身星模型之间直接关联的文献，同时利用ZTF SN Ia DR2和Open Supernova Catalog等公开数据集构建标准化数据集。拒绝因访问受限或内容不相关而无法提取有效信息的条目。 优先选择近五年内发表且包含多波段光度数据、宿主星系属性和前身星模型推断的综合性研究。重点筛选ZTF SN Ia DR2相关论文与Open Supernova Catalog数据集，因其提供大规模、标准化、可复现的Ia型超新星光变曲线数据，支持对峰值星等、衰减率与前身星环境的统计分析。同时保留高被引经典论文作为基准参考。

| Decision | Priority | Role | Provider | Title | Score | Reason |
|---|---|---|---|---|---|---|
| deep_read | high | dataset |  | Open Supernova Catalog | 0.98 | Open Supernova Catalog是Ia型超新星研究的核心开放数据库，集中收录了峰值星等、衰减率（Δm15）、宿主星系、红移等关键参数，支持跨研究比较，直接满足动态提取计划中所有高优先级字段。 |
| deep_read | high | primary_paper | arxiv | Timescale Stretch Parameterization of Type Ia Supernova B-band Light Curves | 0.72 | 该论文提出并验证了光变曲线时间轴拉伸因子（stretch factor）的概念，是理解衰减率与峰值星等关系的奠基性工作，且其方法被广泛用于后续SALT2等模型，对建模与分类具有根本意义。 |
| deep_read | high | primary_paper | arxiv | The Standardizability of Type Ia Supernovae in the Near-Infrared: Evidence for a Peak Luminosity-Decline Rate Relation in the Near-Infrared | 0.72 | 研究近红外波段的标准化问题，发现J/H波段存在显著的峰值亮度-衰减率关系，为提升Ia型超新星作为标准烛光的精度提供关键证据，直接支持用户对衰减率与峰值星等关系的分析需求。 |
| deep_read | high | primary_paper | arxiv | CfA3: 185 Type Ia Supernova Light Curves from the CfA | 0.72 | CfA3是迄今最完整的本地Ia型超新星样本之一，提供了185个事件的同质化多波段测光数据，包含精确的峰值星等与衰减率测量，是构建高质量数据集的理想基础。 |
| deep_read | high | primary_paper | arxiv | ZTF SNe Ia DR2: Towards cosmology-grade ZTF supernova light curves using scene modeling photometry | 0.72 | ZTF SN Ia DR2是目前最大规模的Ia型超新星样本（3628个），其光变曲线数据经过Scene Modeling Photometry处理，具备高精度，是进行大规模统计分析和模型验证的关键数据源。 |
| deep_read | high | primary_paper | arxiv | Multi-Color Light Curves of Type Ia Supernovae on the Color-Magnitude Diagram: a Novel Step Toward More Precise Distance and Extinction Estimates | 0.72 | 首次系统性地提出并验证了“颜色-星等关系”在光变曲线中的线性特性，可提供更精确的距离估计，是改进现有光变曲线拟合方法的重要依据。 |
| deep_read | high | primary_paper | arxiv | An Empirical Fitting Method to Type Ia Supernova Light Curves. III. A Three-Parameter Relationship: Peak Magnitude, Rise Time, and Photospheric Velocity | 0.72 | 提出基于上升时间、峰值速度与峰值星等的三参数关系模型，为理解光变曲线形状与物理起源之间的联系提供了新的物理解释框架。 |
| deep_read | high | primary_paper | arxiv | Model independent bounds on Type Ia supernova absolute peak magnitude | 0.72 | 采用非参数方法对Ia型超新星峰值绝对星等进行模型独立约束，结果稳定在-19.4左右，为统一标准烛光的绝对亮度提供了强有力的支持。 |
| metadata_only | medium | supporting_paper | arxiv | Type Ia Supernovae, the Hubble Constant, the Cosmological Constant, and the Age of the Universe | 0.72 | 早期关于宇宙学常数的开创性论文，虽重要但其核心结论已被后续研究深化，仅作为背景文献保留，无需全文解析。 |
| metadata_only | medium | supporting_paper | arxiv | Time Dilation in the Light Curve of the Distant Type Ia Supernovae SN 1995K | 0.72 | 首次通过时间膨胀验证宇宙膨胀，是历史里程碑，但内容与当前研究目标关联较弱，仅作元数据引用。 |
| metadata_only | medium | supporting_paper | arxiv | Pre-nebular light curves of type I supernovae | 0.72 | 讨论早期光变曲线，但未提供具体数值或可提取参数，仅具理论参考价值。 |
| metadata_only | medium | supporting_paper | arxiv | Light Curve Models for SN 2009dc | 0.72 | 针对超亮Ia型超新星SN 2009dc的模型研究，虽涉及大质量前身星，但样本单一，不具普遍代表性。 |
| metadata_only | medium | supporting_paper | arxiv | The fast declining Type Ia supernova 2003gs, and evidence for a significant dispersion in near-infrared absolute magnitudes of fast decliners at maximum light | 0.72 | 研究快速衰减型Ia型超新星，虽揭示了近红外亮度的双峰分布，但样本量小，结论局部，仅作补充参考。 |
| metadata_only | medium | supporting_paper | arxiv | The absolute infrared magnitudes of type Ia supernovae | 0.72 | 早期关于近红外绝对星等的研究，虽指出IR波段散射小，但未提供具体数值，仅作背景支撑。 |
| metadata_only | medium | supporting_paper | arxiv | Constraining the Properties of SNe~Ia Progenitors from Light Curves | 0.72 | 提出利用主成分分析重建前身星属性，但未公开完整数据，仅作为方法论参考。 |
| metadata_only | medium | supporting_paper | arxiv | Infrared Light Curves of Type Ia Supernovae | 0.72 | 报告近红外光变曲线形态随衰减率的变化，但未提供量化数据，仅作趋势描述。 |
| metadata_only | medium | supporting_paper | arxiv | Artificial Neural Network Spectral Light Curve Template for Type Ia Supernovae and its Cosmological Constraints | 0.72 | 提出人工神经网络光变曲线模板，虽具创新性，但尚未在大规模样本中验证，暂不深入。 |
| metadata_only | medium | supporting_paper | arxiv | The Rise and Fall of Type Ia Supernova Light Curves in the SDSS-II Supernova Survey | 0.72 | 研究SDSS-II样本的上升与下降时间，虽揭示了非对称性，但未提供可提取的数值参数。 |
| metadata_only | medium | supporting_paper | arxiv | Secondary Parameters of Type Ia Supernova Light Curves | 0.72 | 提出两个独立的次级参数控制光变曲线形状，但未给出具体数值，仅作为理论补充。 |
| metadata_only | medium | supporting_paper | arxiv | Can the violent merger of white dwarfs explain the slowest declining Type Ia supernova SN 2011aa? | 0.72 | 研究慢衰减SN 2011aa的暴力合并模型，虽有启发性，但样本唯一，不具普适性。 |

## Source Triage

- Decisions: 1284
- Selected for ingestion: 20
- Actions: download_pdf(20), read_file_manifest(1), read_metadata(49), record_only(7), skip(1207)

| Action | Provider | Title | Score | Cost | Risk | Reason |
|---|---|---|---|---|---|---|
| record_only |  | The Supernova Legacy Survey (SNLS) | 0.95 | unknown | low | SNLS是大型巡天项目，但未提供具体数据链接或文件，仅作背景引用。 |
| record_only |  | Las Cumbres Observatory Global Telescope Network (LCOGT) | 0.9 | unknown | low | LCOGT是望远镜网络，但未提供具体观测数据，仅作技术背景。 |
| read_file_manifest |  | Open Supernova Catalog | 0.98 | unknown | low | Open Supernova Catalog是Ia型超新星研究的核心开放数据库，集中收录了峰值星等、衰减率（Δm15）、宿主星系、红移等关键参数，支持跨研究比较，直接满足动态提取计划中所有高优先级字段。 Executor mapped deep_read to safe data/supplement inspection. |
| record_only |  | NASA Exoplanet Archive - Supernovae Section | 0.75 | unknown | low | NASA Exoplanet Archive含部分空间望远镜数据，但未明确列出Ia型超新星数据集，风险高。 |
| record_only |  | arXiv:astro-ph.HE - 超新星相关论文 | 0.92 | unknown | low | arXiv搜索结果列表本身无具体内容，仅为索引页，无法提取有效数据。 |
| record_only |  | Supplementary Materials of ApJ Papers on SN Ia | 0.94 | unknown | low | 补充材料链接指向单篇论文，但未提供具体文件，无法确认是否包含所需数据。 |
| record_only |  | SIMBAD Astronomical Database | 0.93 | unknown | low | SIMBAD是天文实体知识库，可辅助溯源，但本身不提供光变曲线数据，仅作背景参考。 |
| record_only |  | GitHub - Ia Supernova Light Curve Repository | 0.88 | unknown | low | GitHub仓库包含Ia型超新星光变曲线数据集与Python脚本，适合检查代码流程与数据格式，但需先读README确认可用性。 Executor downgraded read_readme because provider is unknown. |
| read_metadata | arxiv | Type Ia Supernova Models and Progenitor Scenarios | 0.72 | unknown | low | 论文摘要仅描述爆炸机制，未提供具体光变曲线数据或参数，无法提取所需字段。 |
| download_pdf | arxiv | Timescale Stretch Parameterization of Type Ia Supernova B-band Light Curves | 0.72 | medium | low | 该论文提出并验证了光变曲线时间轴拉伸因子（stretch factor）的概念，是理解衰减率与峰值星等关系的奠基性工作，且其方法被广泛用于后续SALT2等模型，对建模与分类具有根本意义。 |
| read_metadata | arxiv | The Standardizability of Type Ia Supernovae in the Near-Infrared: Evidence for a Peak Luminosity-Decline Rate Relation in the Near-Infrared | 0.72 | medium | low | 研究近红外波段的标准化问题，发现J/H波段存在显著的峰值亮度-衰减率关系，为提升Ia型超新星作为标准烛光的精度提供关键证据，直接支持用户对衰减率与峰值星等关系的分析需求。 Deferred by automatic resource cap: max_auto_resources=20. |
| read_metadata | arxiv | CfA3: 185 Type Ia Supernova Light Curves from the CfA | 0.72 | medium | low | CfA3是迄今最完整的本地Ia型超新星样本之一，提供了185个事件的同质化多波段测光数据，包含精确的峰值星等与衰减率测量，是构建高质量数据集的理想基础。 Deferred by automatic resource cap: max_auto_resources=20. |
| read_metadata | arxiv | Pre-nebular light curves of type I supernovae | 0.72 | unknown | low | 讨论早期光变曲线，但未提供具体数值或可提取参数，仅具理论参考价值。 |
| skip | arxiv | Testing Cosmic Acceleration with Type Ia Supernovae | 0.0 | unknown | low | LLM Source Selector did not choose this source for ingestion or metadata reading. |
| read_metadata | arxiv | Time Dilation in the Light Curve of the Distant Type Ia Supernovae SN 1995K | 0.72 | unknown | low | 首次通过时间膨胀验证宇宙膨胀，是历史里程碑，但内容与当前研究目标关联较弱，仅作元数据引用。 |
| skip | arxiv | Probing type Ia supernova properties using bolometric light curves from the Carnegie Supernova Project and the CfA Supernova Group | 0.0 | unknown | low | LLM Source Selector did not choose this source for ingestion or metadata reading. |
| skip | arxiv | Topological Control of Chirality and Spin with Structured Light | 0.72 | unknown | low | 主题为结构光操控，与Ia型超新星完全无关，属于严重误匹配。 |
| read_metadata | arxiv | The light curves of type Ia Supernova 2004fu | 0.72 | unknown | low | 仅提及SN 2004fu的典型光变曲线，未提供具体数值或可提取数据。 |
| read_metadata | arxiv | Type Ia Supernova Progenitors, Environmental Effects, and Cosmic Supernova Rates | 0.72 | unknown | low | 讨论单简并模型与环境效应，但未提供可提取的光变曲线参数。 |
| read_metadata | arxiv | UBVRI Light Curves of 44 Type Ia Supernovae | 0.72 | unknown | low | U-band光度数据虽独特，但未提供具体数值或可提取表格，仅作补充说明。 |

## Paper Structure

- Heading candidates: 1149
- Section blocks: 682
- Section interpreter: LLM
- Section types: abstract(83), analysis(2), background(22), conclusion(9), data(84), discussion(63), experiments(13), introduction(79), method(174), other(13), references(14), related_work(3), results(123)
- Skipped LLM blocks: 34

| Paper | Section | Type | Pages | Confidence |
|---|---|---|---|---|
| arxiv_1207_2695v3_a_search_for_new_candidate_super_chandrasekhar_mass_type_ia_supernovae_in_the_ne.pdf | ABSTRACT | abstract | 1 | 0.9 |
| arxiv_1207_2695v3_a_search_for_new_candidate_super_chandrasekhar_mass_type_ia_supernovae_in_the_ne.pdf | 1. INTRODUCTION | introduction | 1 | 0.95 |
| arxiv_1207_2695v3_a_search_for_new_candidate_super_chandrasekhar_mass_type_ia_supernovae_in_the_ne.pdf | 1. INTRODUCTION | introduction | 2 | 0.95 |
| arxiv_1207_2695v3_a_search_for_new_candidate_super_chandrasekhar_mass_type_ia_supernovae_in_the_ne.pdf | 1. INTRODUCTION | introduction | 3 | 0.95 |
| arxiv_1207_2695v3_a_search_for_new_candidate_super_chandrasekhar_mass_type_ia_supernovae_in_the_ne.pdf | 2.3. Follow-upObservations | method | 3 | 0.85 |
| arxiv_1207_2695v3_a_search_for_new_candidate_super_chandrasekhar_mass_type_ia_supernovae_in_the_ne.pdf | 2.3. Follow-upObservations | method | 4 | 0.85 |
| arxiv_1207_2695v3_a_search_for_new_candidate_super_chandrasekhar_mass_type_ia_supernovae_in_the_ne.pdf | 2.4. Lightcurves | results | 4 | 0.95 |
| arxiv_1207_2695v3_a_search_for_new_candidate_super_chandrasekhar_mass_type_ia_supernovae_in_the_ne.pdf | 2.4. Lightcurves | results | 5 | 0.95 |
| arxiv_1207_2695v3_a_search_for_new_candidate_super_chandrasekhar_mass_type_ia_supernovae_in_the_ne.pdf | 2.4. Lightcurves | results | 6 | 0.95 |
| arxiv_1207_2695v3_a_search_for_new_candidate_super_chandrasekhar_mass_type_ia_supernovae_in_the_ne.pdf | SNfactoryObservationsofCandidateSuper-ChandraSNeIa 7 | data | 7 | 0.85 |
| arxiv_1207_2695v3_a_search_for_new_candidate_super_chandrasekhar_mass_type_ia_supernovae_in_the_ne.pdf | 3. ANALYSIS | analysis | 7 | 0.95 |
| arxiv_1207_2695v3_a_search_for_new_candidate_super_chandrasekhar_mass_type_ia_supernovae_in_the_ne.pdf | 3. ANALYSIS | analysis | 8 | 0.95 |

## Paper Survey

| Paper | Methods | Baselines | Datasets / Objects | Metrics | Records |
|---|---|---|---|---|---|
| Diversity of Decline-Rate-Corrected Type Ia Supernova Rise Times: One Mode or Two? | Diversity of Decline-Rate-Corrected Type Ia Supernova Rise Times; aquaa algorithm and simple parabolic fits to early data |  | SN 1998aq | difference_in_exploded_time; rms_difference_in_exploded_time | 2 |
| Explosion Models for Type Ia Supernovae: A Comparison with Observed Light Curves, distances, H_o and q_o | One-dimensional Lagrangian hydrodynamics with artificial viscosity (Khokhlov, 1991ab); radiation-hydro codes with nuclear networks (Höflich et al. 1995b); piecewise parabolic method (Collela and Woodward 1984); implicit radiation transport ... |  | 37 SNe Ia parameterized explosion models | accuracy of energy release | 1 |
| ZTF SNe Ia DR2: Towards cosmology-grade ZTF supernova light curves using scene modeling photometry | ZTF SNe Ia DR2; The scene modeling method |  |  | parameter_vector_size; decay_rate | 2 |
| The Rising Light Curves of Type Ia Supernovae | Power-law fitting to early light curve data |  |  | rise_time | 3 |
| A COLIBRI Photometric Study of SN 2025bvm: A Normal, Slowly Declining Type Ia Supernova | A COLIBRI Photometric Study of SN 2025bvm; photometric study using COLIBRI telescope observations |  | SN 2025bvm | decay_rate | 1 |
| Probing the Diversity of Type Ia Supernova Light Curves in the Open Supernova Catalog |  |  |  | accuracy | 2 |

## Dynamic Tables Preview

### cross_survey_comparisons

- supernova_name=; metric_name=; source_1=; value_1=; source_2=; value_2= (source: arxiv_1010_5786v3_the_hubble_space_telescope_cluster_supernova_survey_ii_the_type_ia_supernova_rat.pdf, page: 3)
- supernova_name=SN 1998aq; metric_name=rise_time; source_1=aquaa algorithm; value_1=; source_2=simple parabolic fits; value_2= (source: arxiv_0705_0726v2_diversity_of_decline_rate_corrected_type_ia_supernova_rise_times_one_mode_or_two.pdf, page: 7)
- supernova_name=; metric_name=SALT2 stretch (x1); source_1=Rigault et al. (2024,b); value_1=; source_2=Fig. 5; value_2= (source: arxiv_2409_04346v2_ztf_sn_ia_dr2_overview.pdf, page: 7)
- supernova_name=; metric_name=standardized_Hubble_residual_scatter; source_1=ZTF DR2 (Ginolin et al. 2024,a); value_1=0.165; source_2=Brout et al. 2022; value_2=0.15 (source: arxiv_2409_04346v2_ztf_sn_ia_dr2_overview.pdf, page: 7)
- supernova_name=; metric_name=; source_1=; value_1=; source_2=; value_2= (source: arxiv_9602025v1_explosion_models_for_type_ia_supernovae_a_comparison_with_observed_light_curves_.pdf, page: 7)

### light_curve_fits

- supernova_name=SN 2025bvm; fit_method=SALT2; fit_parameters=mmax=15.8, tmax=+0 days, stretch=1.0; residual_std_dev=; chi_squared= (source: arxiv_2601_07745v2_a_colibri_photometric_study_of_sn_2025bvm_a_normal_slowly_declining_type_ia_supe.pdf, page: 2)
- supernova_name=; fit_method=; fit_parameters=; residual_std_dev=; chi_squared= (source: arxiv_1010_5786v3_the_hubble_space_telescope_cluster_supernova_survey_ii_the_type_ia_supernova_rat.pdf, page: 3)
- supernova_name=SN2018fhw; fit_method=Power law fit; fit_parameters=power law exponent; residual_std_dev=; chi_squared= (source: arxiv_1904_02171v2_early_time_light_curves_of_type_ia_supernovae_observed_with_tess.pdf, page: 8)
- supernova_name=SN2018hss; fit_method=not specified; fit_parameters=; residual_std_dev=; chi_squared= (source: arxiv_1904_02171v2_early_time_light_curves_of_type_ia_supernovae_observed_with_tess.pdf, page: 8)
- supernova_name=; fit_method=power law; fit_parameters=A, B, β, t0; residual_std_dev=; chi_squared= (source: arxiv_1904_02171v2_early_time_light_curves_of_type_ia_supernovae_observed_with_tess.pdf, page: 8)

### light_curve_metrics

- supernova_name=SN 2025bvm; redshift=; peak_magnitude=15.8; filter_band=r′; decay_rate=; observation_telescope= (source: arxiv_2601_07745v2_a_colibri_photometric_study_of_sn_2025bvm_a_normal_slowly_declining_type_ia_supe.pdf, page: 2)
- supernova_name=; redshift=; peak_magnitude=; filter_band=; decay_rate=; observation_telescope=Hubble Space Telescope (source: arxiv_1010_5786v3_the_hubble_space_telescope_cluster_supernova_survey_ii_the_type_ia_supernova_rat.pdf, page: 3)
- supernova_name=SN2018fhw; redshift=; peak_magnitude=; filter_band=; decay_rate=; observation_telescope=TESS (source: arxiv_1904_02171v2_early_time_light_curves_of_type_ia_supernovae_observed_with_tess.pdf, page: 8)
- supernova_name=SN2018hss; redshift=; peak_magnitude=; filter_band=; decay_rate=; observation_telescope=TESS (source: arxiv_1904_02171v2_early_time_light_curves_of_type_ia_supernovae_observed_with_tess.pdf, page: 8)
- supernova_name=; redshift=; peak_magnitude=; filter_band=; decay_rate=; observation_telescope= (source: arxiv_1411_1064v1_the_rising_light_curves_of_type_ia_supernovae.pdf, page: 4)

### other_required_fields

- paper_title=A Colibri Photometric Study of SN 2025bvm: A Normal Slowly Declining Type Ia Supernova; material=multi-band optical light curves from Figure 2; method=SNooPy2 and SALT2 via sncosmo; metric_value=15.8; unit=mag; condition=r′-band peak magnitude (source: arxiv_2601_07745v2_a_colibri_photometric_study_of_sn_2025bvm_a_normal_slowly_declining_type_ia_supe.pdf, page: 2)
- paper_title=The Hubble Space Telescope Cluster Supernova Survey II: The Type Ia Supernova Rate; material=HST Cluster Supernova Survey; method=rate calculation based on supernova selection and efficiency studies; metric_value=; unit=; condition=0.9 < z < 1.46 clusters (source: arxiv_1010_5786v3_the_hubble_space_telescope_cluster_supernova_survey_ii_the_type_ia_supernova_rat.pdf, page: 3)
- paper_title=The Hubble Space Telescope Cluster Supernova Survey II: The Type Ia Supernova Rate in High-Redshift Galaxy Clusters; material=HST Cluster SN Survey program; method=Rolling search targeting 25 massive galaxy clusters between July 2005 and December 2006; metric_value=; unit=; condition=Redshift range 0.9 < z < 1.46; clusters selected from X-ray, optical and IR surveys (source: arxiv_1010_5786v3_the_hubble_space_telescope_cluster_supernova_survey_ii_the_type_ia_supernova_rat.pdf, page: 3)
- paper_title=Early Time Light Curves of Type Ia Supernovae Observed with TESS; material=Appendix B; method=Detrending via contaminating star light curve extraction, median filtering, and scaling/shift fitting; metric_value=flux relative to peak; unit=relative; condition=Near bright star causing residual contamination in difference images (source: arxiv_1904_02171v2_early_time_light_curves_of_type_ia_supernovae_observed_with_tess.pdf, page: 8)
- paper_title=Early-time Light Curves of Type Ia Supernovae Observed with TESS; material=Appendix A; method=bootstrap resampling; metric_value=β; unit=dimensionless; condition=fitted between 0.5 and 4.0 (source: arxiv_1904_02171v2_early_time_light_curves_of_type_ia_supernovae_observed_with_tess.pdf, page: 8)

### paper_metadata

- title=Timescale Stretch Parameterization of Type Ia Supernova B-band Light Curves; authors=G. Goldhaber, D. E. Groom, A. Kim, G. Aldering, P. Astier, A. Conley, S. E. Deustua, R. Ellis, S. Fabbro, A. S. Fruchter, A. Goobar, I. Hook, M. Irwin, M. Kim, R. A. Knop, C. Lidman, R. McMahon, P. E. Nugent, R. Pain, N. Panagia, C. R. Pennypacker, S. Perlmutter, P. Ruiz-Lapuente, B. Schaefer, N. A. Walton, T. York, The Supernova Cosmology Project; publication_date=2001-04-24; venue=arXiv:0104382v1 (source: arxiv_0104382v1_timescale_stretch_parameterization_of_type_ia_supernova_b_band_light_curves.pdf, page: None)
- title=Light Curve Models for SN 2009dc; authors=Yasuomi Kamiya; publication_date=2013-02-14; venue=arXiv:1302.3375v1 (source: arxiv_1302_3375v1_light_curve_models_for_sn_2009dc.pdf, page: None)
- title=A COLIBRI Photometric Study of SN 2025bvm: A Normal, Slowly Declining Type Ia Supernova; authors=Diego Hernando Gonzalez-Buitrago, Maria Teresa Garcia-Diaz, Alberto Emiliano Montoya-Olivo, Sergio Sanchez-Sanjuan, Hector Avila-Mogollon; publication_date=2026-01-12; venue=arXiv:2601.07745v2 (source: arxiv_2601_07745v2_a_colibri_photometric_study_of_sn_2025bvm_a_normal_slowly_declining_type_ia_supe.pdf, page: None)
- title=Constraining the Properties of SNe~Ia Progenitors from Light Curves; authors=B. Sadler, P. Hoeflich, E. Baron, K. Krisciunas, G. Folatelli, M. Hamuy, M. Khokhlov. M. Phillips, N. Suntzeff, L. Wang; publication_date=2011-09-16; venue=arXiv:1109.3629v1 (source: arxiv_1109_3629v1_constraining_the_properties_of_sne_ia_progenitors_from_light_curves.pdf, page: None)
- title=ZTF SNe Ia DR2: Towards cosmology-grade ZTF supernova light curves using scene modeling photometry; authors=L. Lacroix, N. Regnault, T. de Jaeger, M. Le Jeune, M. Betoule, J. -M. Colley, M. Bernard, M. Rigault, M. Smith, A. Goobar, K. Maguire, G. Dimitriadis, J. Nordin, J. Johansson, M. Aubert, C. Barjou, E. C. Bellm, S. Bongard, U. Burgaz, B. Carreres, D. Fouchez, F. Feinstein, L. Galbany, M. Ginolin, M. Graham, D. Kuhn, R. R. Laher, T. E. Müller-Bravo, J. Neveu, M. Osman, B. Popovic, B. Racine, P. Rosnet, D. Rosselli, R. Smith, J. Sollerman, J. H. Terwel, A. Townsend, A. Wold; publication_date=2025-09-04; venue=arXiv:2509.04073v1 (source: arxiv_2509_04073v1_ztf_sne_ia_dr2_towards_cosmology_grade_ztf_supernova_light_curves_using_scene_mo.pdf, page: None)

### progenitor_evidence

- supernova_name=SN 2025bvm; progenitor_model=no clear evidence (source: arxiv_2601_07745v2_a_colibri_photometric_study_of_sn_2025bvm_a_normal_slowly_declining_type_ia_supe.pdf, page: 2)
- supernova_name=; progenitor_model= (source: arxiv_1010_5786v3_the_hubble_space_telescope_cluster_supernova_survey_ii_the_type_ia_supernova_rat.pdf, page: 3)
- supernova_name=SN2018fhw; progenitor_model=no clear evidence (source: arxiv_1904_02171v2_early_time_light_curves_of_type_ia_supernovae_observed_with_tess.pdf, page: 8)
- supernova_name=SN2018hss; progenitor_model=no clear evidence (source: arxiv_1904_02171v2_early_time_light_curves_of_type_ia_supernovae_observed_with_tess.pdf, page: 8)
- supernova_name=; progenitor_model= (source: arxiv_9602025v1_explosion_models_for_type_ia_supernovae_a_comparison_with_observed_light_curves_.pdf, page: 7)

## Figure & Chart Extraction

- Figures detected: 293
- Charts extracted (Qwen-VL): 0
- Charts needing review: 0

| Figure | Source | Page | Type | Series | Points | Confidence | Validation |
|---|---|---|---|---|---|---|---|

## Quality

- Issues: 547
- Warnings: 546
- Errors: 0
- Conflicts: 0
- Evidence coverage: 1.0
- Value evidence coverage: 0.6364

## Output Guide

- `tables/*.csv`: dynamic tables generated from the task-specific schema.
- `figures/*.png`: figure regions rendered from PDFs for chart extraction.
- `chart_extractions.json`: VL-extracted chart axes, legends, and data points with figure provenance.
- `chart_data/chart_data_index.csv` + `chart_data/chart_data_*.csv`: long-format chart data points (approximate).
- `chart_validation_report.json`: deterministic axis/series/unit checks for every extracted chart.
- `dynamic_schema.json`: the LLM-generated extraction schema for this task.
- `multi_source_search_plan.json`: the LLM-generated plan for arXiv, OpenAlex, Semantic Scholar, Crossref, Zenodo, Figshare, and GitHub.
- `connector_status.csv`: per-connector success/failure table with query and error messages.
- `discovered_sources.csv`: normalized multi-source search results with provider, URL, query, and metadata.
- `source_selection.csv`: LLM source-selection decisions before executable safety triage.
- `source_selection_plan.json`: full LLM source-selection plan with time range and reasons.
- `source_triage.csv`: source-level ingestion decisions, including why each source was downloaded, deferred, or kept as metadata.
- `source_research.csv`: source-level research evidence collected from metadata, README files, file manifests, and downloaded files.
- `arxiv_search_plan.json`: the LLM-generated arXiv search queries and selection criteria.
- `dynamic_records_clean.json`: cleaned and merged dynamic records with evidence.
- `dynamic_records_raw.json`: raw LLM dynamic records before curation.
- `needs_review.csv`: records or fields flagged by deterministic quality rules.
- `paper_survey.csv`: paper-level survey summary.
- `result.csv`: metric-oriented scientific records.
- `quality_report.json`: evidence, warning, error, and conflict checks.
- `section_plan.json`: heading candidates, LLM section interpretation, and section-aware chunks.
