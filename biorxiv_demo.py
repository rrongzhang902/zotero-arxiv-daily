import arxiv
import argparse
import os
import sys
import time
import yaml
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange,tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper, BiorxivPaper
from llm import set_global_llm
from journal import get_journal_paper
import feedparser
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BIORXIV_CONNECT_TIMEOUT_SECONDS = 30
BIORXIV_READ_TIMEOUT_SECONDS = 120
BIORXIV_MAX_RETRIES = 8
BIORXIV_BACKOFF_FACTOR = 1.5
ARXIV_BATCH_SIZE = 20
ARXIV_BATCH_PAUSE_SECONDS = 3
ARXIV_MAX_RETRIES = 5
ARXIV_RETRY_DELAY_SECONDS = 15


def _normalize_biorxiv_category(category: str) -> str:
    return category.strip().lower().replace(" ", "_").replace("-", "_")

def _build_search_query(arxiv_query: str) -> str:
    cleaned = [q.strip() for q in arxiv_query.replace(" ", "").split("+") if q.strip()]
    if not cleaned:
        return ""
    return " OR ".join(f"cat:{q}" for q in cleaned)


def _is_arxiv_rate_limit_error(exc: Exception) -> bool:
    return "429" in str(exc)


def _fetch_arxiv_batch(client: arxiv.Client, batch_ids: list[str]) -> list[ArxivPaper]:
    delay = ARXIV_RETRY_DELAY_SECONDS
    last_exc = None
    for attempt in range(ARXIV_MAX_RETRIES):
        try:
            search = arxiv.Search(id_list=batch_ids)
            return [ArxivPaper(p) for p in client.results(search)]
        except Exception as exc:
            last_exc = exc
            if not _is_arxiv_rate_limit_error(exc):
                raise
            logger.warning(
                "arXiv rate limited for batch size {} (attempt {}/{}). Retrying in {}s.",
                len(batch_ids),
                attempt + 1,
                ARXIV_MAX_RETRIES,
                delay,
            )
            time.sleep(delay)
            delay *= 2

    if len(batch_ids) <= 5:
        raise last_exc

    mid = len(batch_ids) // 2
    logger.warning(
        "Repeated arXiv 429 for batch size {}. Splitting batch into {} and {}.",
        len(batch_ids),
        mid,
        len(batch_ids) - mid,
    )
    left = _fetch_arxiv_batch(client, batch_ids[:mid])
    time.sleep(ARXIV_BATCH_PAUSE_SECONDS)
    right = _fetch_arxiv_batch(client, batch_ids[mid:])
    return left + right

def get_zotero_corpus(id:str,key:str) -> list[dict]:
    zot = zotero.Zotero(id, 'user', key)
    collections = zot.everything(zot.collections())
    collections = {c['key']:c for c in collections}
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    corpus = [c for c in corpus if c['data']['abstractNote'] != '']
    def get_collection_path(col_key:str) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']
    for c in corpus:
        paths = [get_collection_path(col) for col in c['data']['collections']]
        c['paths'] = paths
    return corpus

def filter_corpus(corpus:list[dict], pattern:str) -> list[dict]:
    _,filename = mkstemp()
    with open(filename,'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename,base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus


def get_arxiv_paper(query:str, debug:bool=False) -> list[ArxivPaper]:
    if query is None or query.strip() == "":
        logger.info("No arXiv query configured.")
        return []
    client = arxiv.Client(num_retries=10,delay_seconds=10)
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if 'Feed error for query' in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
    if not debug:
        papers = []
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.arxiv_announce_type == 'new']
        if len(all_paper_ids) == 0:
            logger.info("No new arXiv papers found today. Fetching the most recent submissions instead.")
            search_query = _build_search_query(query)
            if search_query:
                search = arxiv.Search(query=search_query, sort_by=arxiv.SortCriterion.SubmittedDate, max_results=50)
                return [ArxivPaper(p) for p in client.results(search)]
            return []
        bar = tqdm(total=len(all_paper_ids),desc="Retrieving Arxiv papers")
        for i in range(0,len(all_paper_ids),ARXIV_BATCH_SIZE):
            batch_ids = all_paper_ids[i:i+ARXIV_BATCH_SIZE]
            batch = _fetch_arxiv_batch(client, batch_ids)
            bar.update(len(batch))
            papers.extend(batch)
            time.sleep(ARXIV_BATCH_PAUSE_SECONDS)
        bar.close()

    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(query='cat:cs.AI', sort_by=arxiv.SortCriterion.SubmittedDate)
        papers = []
        for i in client.results(search):
            papers.append(ArxivPaper(i))
            if len(papers) == 5:
                break

    return papers

def get_biorxiv_paper(query: str, debug: bool = False) -> list[BiorxivPaper]:
    """
    Retrieve papers from bioRxiv API.
    
    Args:
        query: The search query/category
        debug: Whether to run in debug mode
        
    Returns:
        A list of BiorxivPaper objects
        
    Raises:
        requests.RequestException: If the API request fails
        ValueError: If the query is invalid
    """
    if query is None or query.strip() == "":
        logger.info("No bioRxiv query configured.")
        return []

    session = requests.Session()
    retries = Retry(
        total=BIORXIV_MAX_RETRIES,
        connect=BIORXIV_MAX_RETRIES,
        read=BIORXIV_MAX_RETRIES,
        status=BIORXIV_MAX_RETRIES,
        backoff_factor=BIORXIV_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    if not debug:
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        formatted_date = today.strftime("%Y-%m-%d")
        formatted_yesterday = yesterday.strftime("%Y-%m-%d")
        if "+" in query:
            queries = query.split("+")
        else:
            queries = [query]

        papers = []
        failed_queries = []
        base_url = f"https://api.biorxiv.org/details/biorxiv/{formatted_yesterday}/{formatted_date}"
        for raw_query in queries:
            normalized_query = _normalize_biorxiv_category(raw_query)
            logger.info(
                "Retrieving biorxiv papers from {} with category={}...",
                base_url,
                normalized_query,
            )
            try:
                response = session.get(
                    base_url,
                    params={"category": normalized_query},
                    timeout=(
                        BIORXIV_CONNECT_TIMEOUT_SECONDS,
                        BIORXIV_READ_TIMEOUT_SECONDS,
                    ),
                )
                if response.status_code != 200:
                    logger.warning(
                        "bioRxiv request failed for category={} with status={} url={}",
                        normalized_query,
                        response.status_code,
                        response.url,
                    )
                    failed_queries.append(normalized_query)
                    continue
                data = response.json()
            except requests.RequestException as exc:
                logger.warning(
                    "bioRxiv request errored for category={} with {}",
                    normalized_query,
                    exc,
                )
                failed_queries.append(normalized_query)
                continue
            except ValueError as exc:
                logger.warning(
                    "bioRxiv returned invalid JSON for category={} with {}",
                    normalized_query,
                    exc,
                )
                failed_queries.append(normalized_query)
                continue

            for i in data['collection']:
                if i['doi'] == '':
                    continue
                paper = BiorxivPaper(i)
                papers.append(paper)
        if failed_queries:
            logger.warning(
                "Skipped {} bioRxiv categories due to API failures: {}",
                len(failed_queries),
                ", ".join(failed_queries),
            )
    else:
        url = "https://api.biorxiv.org/details/biorxiv/2025-03-21/2025-03-28"
        response = session.get(
            url,
            params={"category": "cell_biology"},
            timeout=(
                BIORXIV_CONNECT_TIMEOUT_SECONDS,
                BIORXIV_READ_TIMEOUT_SECONDS,
            ),
        )
        if response.status_code != 200:
            raise Exception(
                f"bioRxiv debug request failed with status={response.status_code}, url={response.url}."
            )
        data = response.json()
        logger.debug("Retrieve 5 biorxiv papers regardless of the date.")
        papers = []
        for i in data['collection']:
            if i['doi'] == '':
                continue
            paper = BiorxivPaper(i)
            papers.append(paper)
            if len(papers) == 5:
                break
    return papers


parser = argparse.ArgumentParser(description='Recommender system for academic papers')

def add_argument(*args, **kwargs):
    def get_env(key:str,default=None):
        # handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == '' or v is None:
            return default
        return v
    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get('dest',args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        #convert env_value to the specified type
        if kwargs.get('type') == bool:
            env_value = env_value.lower() in ['true','1']
        else:
            env_value = kwargs.get('type')(env_value)
        parser.set_defaults(**{arg_full_name:env_value})


if __name__ == '__main__':
    
    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email',default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=50)
    add_argument('--max_biorxiv_num', type=int, help='Maximum number of biorxiv papers to recommend',default=50)
    add_argument('--max_journal_num', type=int, help='Maximum number of journal papers to recommend',default=50)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')
    add_argument('--biorxiv_query', type=str, help='Biorxiv search category')
    add_argument('--journal_sources', type=str, help='Configured journal sources')
    add_argument('--journal_group', type=str, help='Configured journal group', default='all')
    add_argument('--journal_lookback_days', type=int, help='Lookback days for journal PubMed entry window', default=1)
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Target language for TLDR translation",
        default="Chinese",
    )
    add_argument(
        "--use_volcengine_translation",
        type=bool,
        help="Use Volcengine for TLDR translation",
        default=True,
    )
    add_argument(
        "--volcengine_api_key",
        type=str,
        help="Volcengine API key for translation",
        default=None,
    )
    add_argument(
        "--volcengine_base_url",
        type=str,
        help="Volcengine translation endpoint",
        default="https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    )
    add_argument(
        "--volcengine_translation_model",
        type=str,
        help="Volcengine model for translation",
        default="doubao-seed-2-0-lite-260215",
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    # load arguments from .yml file
    if os.path.exists("config.yml"):
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
            # load arguments to args
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
            
    assert (
        not args.use_llm_api or args.openai_api_key is not None
    )  # If use_llm_api is True, openai_api_key must be provided
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")
    logger.info("Retrieving candidate papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)
    biorxiv_papers = get_biorxiv_paper(args.biorxiv_query, args.debug)
    journal_papers = get_journal_paper(
        args.journal_sources,
        journal_group=args.journal_group,
        debug=args.debug,
        lookback_days=args.journal_lookback_days,
    )
    logger.info(f"Retrieved {len(papers)} papers from Arxiv.")
    logger.info(f"Retrieved {len(biorxiv_papers)} papers from Biorxiv.")
    logger.info(f"Retrieved {len(journal_papers)} papers from Journals.")
    if len(papers) == 0 and len(biorxiv_papers) == 0 and len(journal_papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
        if not args.send_empty:
          exit(0)
    else:
        logger.info("Reranking papers...")
        papers, biorxiv_papers, journal_papers = rerank_paper(papers, biorxiv_papers, journal_papers, corpus)
        if args.max_paper_num != -1 and args.max_paper_num < len(papers):
            papers = papers[:args.max_paper_num]
        if args.max_biorxiv_num != -1 and args.max_biorxiv_num < len(biorxiv_papers):
            biorxiv_papers = biorxiv_papers[:args.max_biorxiv_num]
        if args.max_journal_num != -1 and args.max_journal_num < len(journal_papers):
            journal_papers = journal_papers[:args.max_journal_num]
        if args.use_llm_api:
            logger.info("Using OpenAI API as global LLM.")
            set_global_llm(
                api_key=args.openai_api_key,
                base_url=args.openai_api_base,
                model=args.model_name,
                lang=args.language,
                use_volcengine_translation=args.use_volcengine_translation,
                volcengine_api_key=args.volcengine_api_key,
                volcengine_base_url=args.volcengine_base_url,
                volcengine_translation_model=args.volcengine_translation_model,
            )
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(
                lang=args.language,
                use_volcengine_translation=args.use_volcengine_translation,
                volcengine_api_key=args.volcengine_api_key,
                volcengine_base_url=args.volcengine_base_url,
                volcengine_translation_model=args.volcengine_translation_model,
            )

    html = render_email(papers, biorxiv_papers, journal_papers)
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")
