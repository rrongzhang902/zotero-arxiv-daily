from dataclasses import dataclass
import re
import xml.etree.ElementTree as ET

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from paper import JournalPaper

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_CONNECT_TIMEOUT_SECONDS = 20
PUBMED_READ_TIMEOUT_SECONDS = 60
PUBMED_MAX_RETRIES = 5
PUBMED_BACKOFF_FACTOR = 1.0
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_FETCH_PER_JOURNAL = 10


@dataclass(frozen=True)
class JournalConfig:
    key: str
    name: str
    pubmed_query: str


SUPPORTED_JOURNALS: list[JournalConfig] = [
    JournalConfig("nature", "Nature", "Nature"),
    JournalConfig("science", "Science", "Science"),
    JournalConfig("cell", "Cell", "Cell"),
    JournalConfig("pnas", "PNAS", "Proceedings of the National Academy of Sciences of the United States of America"),
    JournalConfig("nature_biotechnology", "Nature Biotechnology", "Nature Biotechnology"),
    JournalConfig("nature_methods", "Nature Methods", "Nature Methods"),
    JournalConfig("nature_chemical_biology", "Nature Chemical Biology", "Nature Chemical Biology"),
    JournalConfig("nature_structural_molecular_biology", "Nature Structural & Molecular Biology", "Nature Structural & Molecular Biology"),
    JournalConfig("nature_machine_intelligence", "Nature Machine Intelligence", "Nature Machine Intelligence"),
    JournalConfig("nature_computational_science", "Nature Computational Science", "Nature Computational Science"),
    JournalConfig("science_advances", "Science Advances", "Science Advances"),
    JournalConfig("cell_systems", "Cell Systems", "Cell Systems"),
    JournalConfig("cell_genomics", "Cell Genomics", "Cell Genomics"),
    JournalConfig("neuron", "Neuron", "Neuron"),
    JournalConfig("patterns", "Patterns", "Patterns (N Y)"),
    JournalConfig("ajhg", "American Journal of Human Genetics", "American Journal of Human Genetics"),
    JournalConfig("trends_in_genetics", "Trends in Genetics", "Trends in Genetics"),
    JournalConfig("bioinformatics", "Bioinformatics", "Bioinformatics"),
    JournalConfig("briefings_in_bioinformatics", "Briefings in Bioinformatics", "Briefings in Bioinformatics"),
    JournalConfig("nucleic_acids_research", "Nucleic Acids Research", "Nucleic Acids Research"),
    JournalConfig("genome_biology", "Genome Biology", "Genome Biology"),
    JournalConfig("genome_research", "Genome Research", "Genome Research"),
    JournalConfig("genome_medicine", "Genome Medicine", "Genome Medicine"),
    JournalConfig("nature_communications", "Nature Communications", "Nature Communications"),
    JournalConfig("nature_genetics", "Nature Genetics", "Nature Genetics"),
    JournalConfig("genetics", "GENETICS", "Genetics"),
    JournalConfig("human_molecular_genetics", "Human Molecular Genetics", "Human Molecular Genetics"),
    JournalConfig("genetics_in_medicine", "Genetics in Medicine", "Genetics in Medicine"),
    JournalConfig("nature_reviews_genetics", "Nature Reviews Genetics", "Nature Reviews Genetics"),
    JournalConfig("brain", "Brain", "Brain"),
    JournalConfig("american_journal_of_psychiatry", "American Journal of Psychiatry", "American Journal of Psychiatry"),
    JournalConfig("nature_neuroscience", "Nature Neuroscience", "Nature Neuroscience"),
    JournalConfig("molecular_psychiatry", "Molecular Psychiatry", "Molecular Psychiatry"),
    JournalConfig("biological_psychiatry", "Biological Psychiatry", "Biological Psychiatry"),
    JournalConfig("translational_psychiatry", "Translational Psychiatry", "Translational Psychiatry"),
    JournalConfig("jama_psychiatry", "JAMA Psychiatry", "JAMA Psychiatry"),
    JournalConfig("protein_engineering_design_and_selection", "Protein Engineering, Design and Selection", "Protein Engineering Design and Selection"),
    JournalConfig("protein_science", "Protein Science", "Protein Science"),
    JournalConfig("structure", "Structure", "Structure"),
    JournalConfig("journal_of_molecular_biology", "Journal of Molecular Biology", "Journal of Molecular Biology"),
]


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


JOURNAL_ALIASES: dict[str, JournalConfig] = {}
for cfg in SUPPORTED_JOURNALS:
    aliases = {
        cfg.key,
        cfg.name,
        cfg.name.replace("&", "and"),
    }
    for alias in aliases:
        JOURNAL_ALIASES[_normalize_token(alias)] = cfg


JOURNAL_GROUPS: dict[str, list[str]] = {
    "all": [cfg.key for cfg in SUPPORTED_JOURNALS],
    "xx": [
        "nature",
        "science",
        "cell",
        "pnas",
        "nature_biotechnology",
        "nature_methods",
        "nature_chemical_biology",
        "nature_structural_molecular_biology",
        "nature_machine_intelligence",
        "nature_computational_science",
        "science_advances",
        "cell_systems",
        "bioinformatics",
        "briefings_in_bioinformatics",
        "nucleic_acids_research",
        "nature_communications",
    ],
    "rr": [
        "nature",
        "science",
        "cell",
        "pnas",
        "nature_methods",
        "science_advances",
        "cell_genomics",
        "neuron",
        "ajhg",
        "trends_in_genetics",
        "bioinformatics",
        "genome_biology",
        "genome_research",
        "genome_medicine",
        "nature_communications",
        "nature_genetics",
        "genetics",
        "human_molecular_genetics",
        "genetics_in_medicine",
        "brain",
        "american_journal_of_psychiatry",
        "nature_neuroscience",
        "molecular_psychiatry",
        "biological_psychiatry",
        "translational_psychiatry",
        "jama_psychiatry",
    ],
}


def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=PUBMED_MAX_RETRIES,
        connect=PUBMED_MAX_RETRIES,
        read=PUBMED_MAX_RETRIES,
        status=PUBMED_MAX_RETRIES,
        backoff_factor=PUBMED_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _configs_from_group(group: str) -> list[JournalConfig]:
    group_key = _normalize_token(group or "all") or "all"
    if group_key not in JOURNAL_GROUPS:
        logger.warning("Unknown journal group '{}'. Falling back to all.", group)
        group_key = "all"
    group_keys = set(JOURNAL_GROUPS[group_key])
    return [cfg for cfg in SUPPORTED_JOURNALS if cfg.key in group_keys]


def parse_journal_sources(raw: str, group: str = "all") -> list[JournalConfig]:
    if raw is None or raw.strip() == "":
        return _configs_from_group(group)
    tokens = [t.strip() for t in re.split(r"[,\n;+]+", raw) if t.strip()]
    if any(_normalize_token(t) == "all" for t in tokens):
        return _configs_from_group("all")
    selected = []
    seen = set()
    for token in tokens:
        cfg = JOURNAL_ALIASES.get(_normalize_token(token))
        if cfg is None:
            logger.warning("Unknown journal source '{}'. Skipping.", token)
            continue
        if cfg.key in seen:
            continue
        seen.add(cfg.key)
        selected.append(cfg)
    return selected


def _pubmed_search(
    session: requests.Session,
    config: JournalConfig,
    lookback_days: int,
    retmax: int,
    debug: bool,
) -> list[str]:
    params = {
        "db": "pubmed",
        "term": f"{config.pubmed_query}[Journal] AND hasabstract[text]",
        "retmax": retmax,
        "retmode": "json",
        "sort": "pub date",
    }
    if not debug:
        params["reldate"] = lookback_days
        params["datetype"] = "pdat"
    response = session.get(
        PUBMED_SEARCH_URL,
        params=params,
        timeout=(PUBMED_CONNECT_TIMEOUT_SECONDS, PUBMED_READ_TIMEOUT_SECONDS),
    )
    response.raise_for_status()
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _pubmed_fetch(session: requests.Session, ids: list[str]) -> ET.Element:
    response = session.get(
        PUBMED_FETCH_URL,
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        timeout=(PUBMED_CONNECT_TIMEOUT_SECONDS, PUBMED_READ_TIMEOUT_SECONDS),
    )
    response.raise_for_status()
    return ET.fromstring(response.text)


def _text_from_element(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _parse_abstract(article: ET.Element) -> str:
    texts = []
    for abstract_text in article.findall(".//Abstract/AbstractText"):
        label = abstract_text.attrib.get("Label")
        content = _text_from_element(abstract_text)
        if not content:
            continue
        texts.append(f"{label}: {content}" if label else content)
    return "\n".join(texts).strip()


def _parse_authors(article: ET.Element) -> list[str]:
    authors = []
    for author in article.findall(".//AuthorList/Author"):
        collective_name = author.findtext("CollectiveName")
        if collective_name:
            authors.append(collective_name)
            continue
        last_name = author.findtext("LastName")
        fore_name = author.findtext("ForeName")
        if last_name and fore_name:
            authors.append(f"{fore_name} {last_name}")
        elif last_name:
            authors.append(last_name)
    return authors


def _parse_published_at(article: ET.Element) -> str:
    article_date = article.find(".//ArticleDate")
    if article_date is not None:
        year = article_date.findtext("Year")
        month = article_date.findtext("Month")
        day = article_date.findtext("Day")
        if year and month and day:
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    pub_date = article.find(".//PubDate")
    if pub_date is None:
        return ""
    year = pub_date.findtext("Year") or ""
    month = pub_date.findtext("Month") or "01"
    day = pub_date.findtext("Day") or "01"
    month_map = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }
    month = month_map.get(month, month)
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}" if year else ""


def _article_to_paper(article: ET.Element, config: JournalConfig) -> JournalPaper | None:
    title = _text_from_element(article.find(".//ArticleTitle"))
    abstract = _parse_abstract(article)
    if not title or not abstract:
        return None
    authors = _parse_authors(article)
    pmid = _text_from_element(article.find(".//PMID"))
    doi = ""
    for article_id in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        if article_id.attrib.get("IdType") == "doi":
            doi = _text_from_element(article_id)
            break
    paper_url = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    paper_id = doi or pmid
    if not paper_id:
        return None
    return JournalPaper(
        {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "paper_id": paper_id,
            "paper_url": paper_url,
            "journal": config.name,
            "published_at": _parse_published_at(article),
        }
    )


def get_journal_paper(
    journal_sources: str,
    journal_group: str = "all",
    debug: bool = False,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    fetch_per_journal: int = DEFAULT_FETCH_PER_JOURNAL,
) -> list[JournalPaper]:
    configs = parse_journal_sources(journal_sources, group=journal_group)
    if len(configs) == 0:
        logger.info("No journal sources configured.")
        return []

    session = _build_session()
    papers = []
    seen = set()
    per_journal_limit = 5 if debug else fetch_per_journal
    for config in configs:
        logger.info("Retrieving journal papers from {}...", config.name)
        try:
            ids = _pubmed_search(session, config, lookback_days, per_journal_limit, debug)
            if len(ids) == 0:
                continue
            root = _pubmed_fetch(session, ids)
        except Exception as exc:
            logger.warning("Failed to retrieve {} with {}", config.name, exc)
            continue
        for article in root.findall(".//PubmedArticle"):
            paper = _article_to_paper(article, config)
            if paper is None:
                continue
            dedupe_key = paper.paper_url.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            papers.append(paper)
    papers.sort(key=lambda paper: paper.published_at or "", reverse=True)
    return papers
