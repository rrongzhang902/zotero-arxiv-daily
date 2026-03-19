import arxiv

def _get_pdf_url_patch(links) -> str:
    """
    Finds the PDF link among a result's links and returns its URL.
    Should only be called once for a given `Result`, in its constructor.
    After construction, the URL should be available in `Result.pdf_url`.
    """
    pdf_urls = [link.href for link in links if "pdf" in link.href]
    if len(pdf_urls) == 0:
        return None
    return pdf_urls[0]

arxiv.Result._get_pdf_url = _get_pdf_url_patch

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
from paper import ArxivPaper
from llm import set_global_llm
import feedparser

ARXIV_BATCH_SIZE = 20
ARXIV_BATCH_PAUSE_SECONDS = 3
ARXIV_MAX_RETRIES = 5
ARXIV_RETRY_DELAY_SECONDS = 15

def _build_search_query(arxiv_query: str) -> str:
    # Convert category list like "cs.AI+cs.CL" to a search query understood by arxiv.Search
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
                search = arxiv.Search(query=search_query, sort_by=arxiv.SortCriterion.SubmittedDate, max_results=20)
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
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')
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
    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)
    if len(papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
        if not args.send_empty:
          exit(0)
    else:
        logger.info("Reranking papers...")
        papers, _, _ = rerank_paper(papers, [], [], corpus)
        if args.max_paper_num != -1:
            papers = papers[:args.max_paper_num]
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

    html = render_email(papers, [], [])
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")
