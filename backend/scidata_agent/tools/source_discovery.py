from __future__ import annotations

from scidata_agent.agent.schemas import DiscoveredSource, SourceDiscoveryPlan


GENERAL_SOURCES = [
    DiscoveredSource(
        title="arXiv",
        source_type="paper_search",
        url="https://arxiv.org",
        description="Open preprint search for scientific papers.",
        reason="适合作为通用论文发现入口。",
        confidence=0.78,
    ),
    DiscoveredSource(
        title="Semantic Scholar",
        source_type="paper_search",
        url="https://www.semanticscholar.org",
        description="General academic literature search and metadata source.",
        reason="适合查找论文、引用和相关研究主题。",
        confidence=0.76,
    ),
    DiscoveredSource(
        title="Crossref",
        source_type="paper_search",
        url="https://www.crossref.org",
        description="DOI and publication metadata discovery.",
        reason="适合查找正式出版论文和 DOI 元数据。",
        confidence=0.68,
    ),
    DiscoveredSource(
        title="Zenodo",
        source_type="open_database",
        url="https://zenodo.org",
        description="Open repository for datasets, software, and supplementary research artifacts.",
        reason="适合查找开放数据集和论文补充数据。",
        confidence=0.72,
    ),
    DiscoveredSource(
        title="Figshare",
        source_type="open_database",
        url="https://figshare.com",
        description="Repository for figures, datasets, tables, and supplementary materials.",
        reason="适合查找图表、表格和补充材料。",
        confidence=0.68,
    ),
    DiscoveredSource(
        title="GitHub",
        source_type="repository",
        url="https://github.com",
        description="Code and dataset repository search.",
        reason="适合查找研究代码、数据处理脚本和公开数据文件。",
        confidence=0.62,
    ),
]


DOMAIN_PROFILES = {
    "astronomy": {
        "keywords": ["light curve", "photometry", "redshift", "survey", "catalog"],
        "sources": [
            ("NASA ADS", "paper_search", "https://ui.adsabs.harvard.edu", "天文学论文和引用检索入口。"),
            ("VizieR", "open_database", "https://vizier.cds.unistra.fr", "天文学机器可读表格和目录数据库。"),
            ("SIMBAD", "open_database", "https://simbad.cds.unistra.fr", "天体对象标识和基础元数据。"),
            ("Transient Name Server", "open_database", "https://www.wis-tns.org", "瞬变源和超新星发现报告。"),
        ],
    },
    "materials science": {
        "keywords": ["material", "synthesis", "stability", "efficiency", "composition"],
        "sources": [
            ("Materials Project", "open_database", "https://materialsproject.org", "材料结构和性质数据库。"),
            ("NOMAD", "open_database", "https://nomad-lab.eu", "材料计算和实验数据基础设施。"),
            ("NIST Materials Data Repository", "open_database", "https://materialsdata.nist.gov", "材料数据仓库。"),
        ],
    },
    "machine learning": {
        "keywords": ["dataset", "benchmark", "leaderboard", "model", "metric"],
        "sources": [
            ("Papers with Code", "open_database", "https://paperswithcode.com", "机器学习论文、数据集、指标和排行榜。"),
            ("Hugging Face", "open_database", "https://huggingface.co", "模型、数据集和评测资源。"),
        ],
    },
    "biomedicine": {
        "keywords": ["assay", "cohort", "gene", "drug", "clinical"],
        "sources": [
            ("PubMed", "paper_search", "https://pubmed.ncbi.nlm.nih.gov", "生物医学论文检索入口。"),
            ("ClinicalTrials.gov", "open_database", "https://clinicaltrials.gov", "临床试验注册和结果数据。"),
            ("GEO", "open_database", "https://www.ncbi.nlm.nih.gov/geo", "基因表达和组学数据。"),
        ],
    },
    "chemistry": {
        "keywords": ["compound", "reaction", "yield", "catalyst", "spectrum"],
        "sources": [
            ("PubChem", "open_database", "https://pubchem.ncbi.nlm.nih.gov", "化合物结构和性质数据。"),
            ("ChemRxiv", "paper_search", "https://chemrxiv.org", "化学预印本检索入口。"),
        ],
    },
    "environmental science": {
        "keywords": ["monitoring", "station", "climate", "pollution", "time series"],
        "sources": [
            ("NOAA Data Search", "open_database", "https://data.noaa.gov", "气象、海洋和环境开放数据。"),
            ("NASA Earthdata", "open_database", "https://www.earthdata.nasa.gov", "地球观测和遥感数据。"),
        ],
    },
}


def fallback_discover_sources(research_question: str) -> SourceDiscoveryPlan:
    domain = infer_domain(research_question)
    keywords = build_keywords(research_question, domain)
    sources = build_candidate_sources(research_question, domain)
    dynamic_schema = build_dynamic_schema(domain)
    target_data_types = ["papers", "tables", "supplementary_materials", "open_databases"]
    if any(term in research_question.lower() for term in ["figure", "image", "chart", "curve", "图", "曲线"]):
        target_data_types.append("images_or_charts")

    return SourceDiscoveryPlan(
        research_goal=research_question,
        domain=domain,
        recommended_keywords=keywords,
        target_data_types=target_data_types,
        dynamic_schema=dynamic_schema,
        candidate_sources=sources,
        notes=[
            "This is a deterministic fallback source discovery plan for local testing.",
            "Official mode should use Qwen Source Discovery to infer domain, schema, and sources.",
        ],
    )


def infer_domain(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["supernova", "ia 型", "光变", "light curve", "photometry", "redshift", "天文"]):
        return "astronomy"
    if any(token in lowered for token in ["perovskite", "pce", "材料", "catalyst", "bandgap", "stability"]):
        return "materials science"
    if any(
        token in lowered
        for token in [
            "model",
            "dataset",
            "benchmark",
            "fid",
            "lpips",
            "accuracy",
            "tryon",
            "try-on",
            "try on",
            "virtual try",
            "vton",
            "虚拟试衣",
            "试穿",
            "机器学习",
        ]
    ):
        return "machine learning"
    if any(token in lowered for token in ["drug", "cell", "gene", "clinical", "ic50", "biomedical", "生物", "药物"]):
        return "biomedicine"
    if any(token in lowered for token in ["compound", "reaction", "yield", "spectrum", "chemistry", "化学"]):
        return "chemistry"
    if any(token in lowered for token in ["pm2.5", "climate", "pollution", "气象", "环境"]):
        return "environmental science"
    return "general science"


def build_keywords(research_question: str, domain: str) -> list[str]:
    base = [research_question]
    profile = DOMAIN_PROFILES.get(domain)
    if profile:
        base.extend(profile["keywords"])
    base.extend(["dataset", "supplementary data", "table", "open database"])
    return list(dict.fromkeys(base))


def build_candidate_sources(research_question: str, domain: str) -> list[DiscoveredSource]:
    sources = [source.model_copy(deep=True) for source in GENERAL_SOURCES]
    profile = DOMAIN_PROFILES.get(domain)
    if profile:
        for title, source_type, url, description in profile["sources"]:
            sources.append(
                DiscoveredSource(
                    title=title,
                    source_type=source_type,
                    url=url,
                    query=research_question,
                    description=description,
                    reason=f"该来源与 {domain} 领域的数据发现高度相关。",
                    confidence=0.82,
                )
            )
    for source in sources:
        if source.query is None:
            source.query = research_question
    return sources


def build_dynamic_schema(domain: str) -> dict[str, str]:
    common = {
        "entity": "string",
        "entity_type": "string",
        "metric_name": "string",
        "metric_value": "number|string|null",
        "unit": "string|null",
        "condition": "string|null",
        "source": "string",
    }
    domain_fields = {
        "astronomy": {
            "object_name": "string",
            "mjd": "number|null",
            "filter": "string|null",
            "magnitude": "number|null",
            "magnitude_error": "number|null",
            "redshift": "number|null",
            "survey": "string|null",
        },
        "materials science": {
            "material": "string",
            "composition": "string|null",
            "synthesis_method": "string|null",
            "stability_hours": "number|null",
            "test_condition": "string|null",
        },
        "machine learning": {
            "model_name": "string",
            "dataset": "string|null",
            "task": "string|null",
            "metric": "string",
            "score": "number|null",
        },
        "biomedicine": {
            "drug_name": "string|null",
            "cell_line": "string|null",
            "assay_type": "string|null",
            "dose": "string|null",
            "species": "string|null",
        },
    }
    common.update(domain_fields.get(domain, {}))
    return common
