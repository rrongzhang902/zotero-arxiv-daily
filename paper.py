from contextlib import ExitStack
from functools import cached_property
from tempfile import TemporaryDirectory
from typing import Optional
from urllib.error import HTTPError
import re
import tarfile

import arxiv
import requests
import tiktoken
from loguru import logger
from requests.adapters import HTTPAdapter, Retry

from llm import get_llm


def _truncate_prompt(prompt: str) -> str:
    enc = tiktoken.encoding_for_model("gpt-4o")
    prompt_tokens = enc.encode(prompt)
    prompt_tokens = prompt_tokens[:4000]
    return enc.decode(prompt_tokens)


def _generate_scientific_tldr(prompt: str) -> str:
    llm = get_llm()
    return llm.generate(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user.",
            },
            {"role": "user", "content": _truncate_prompt(prompt)},
        ]
    )


def _translate_tldr(text: str) -> str:
    llm = get_llm()
    return llm.translate(text)


class ArxivPaper:
    def __init__(self, paper: arxiv.Result):
        self._paper = paper
        self.score = None

    @property
    def title(self) -> str:
        return self._paper.title

    @property
    def summary(self) -> str:
        return self._paper.summary

    @property
    def authors(self) -> list[str]:
        return self._paper.authors

    @cached_property
    def arxiv_id(self) -> str:
        return re.sub(r"v\d+$", "", self._paper.get_short_id())

    @property
    def pdf_url(self) -> str:
        if self._paper.pdf_url is not None:
            return self._paper.pdf_url

        pdf_url = f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"
        if self._paper.links is not None:
            pdf_url = self._paper.links[0].href.replace("abs", "pdf")

        self._paper.pdf_url = pdf_url
        return pdf_url

    @cached_property
    def code_url(self) -> Optional[str]:
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        s.mount("https://", HTTPAdapter(max_retries=retries))
        try:
            paper_list = s.get(
                f"https://paperswithcode.com/api/v1/papers/?arxiv_id={self.arxiv_id}"
            ).json()
        except Exception as e:
            logger.debug(f"Error when searching {self.arxiv_id}: {e}")
            return None

        if paper_list.get("count", 0) == 0:
            return None
        paper_id = paper_list["results"][0]["id"]

        try:
            repo_list = s.get(
                f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/"
            ).json()
        except Exception as e:
            logger.debug(f"Error when searching {self.arxiv_id}: {e}")
            return None
        if repo_list.get("count", 0) == 0:
            return None
        return repo_list["results"][0]["url"]

    @cached_property
    def tex(self) -> Optional[dict[str, str]]:
        with ExitStack() as stack:
            tmpdirname = stack.enter_context(TemporaryDirectory())
            try:
                file = self._paper.download_source(dirpath=tmpdirname)
            except HTTPError as e:
                if e.code == 404:
                    logger.warning(
                        f"Source for {self.arxiv_id} not found (404). Skipping source analysis."
                    )
                    return None
                logger.error(
                    f"HTTP Error {e.code} when downloading source for {self.arxiv_id}: {e.reason}"
                )
                raise
            except Exception as e:
                logger.error(f"Error when downloading source for {self.arxiv_id}: {e}")
                return None
            try:
                tar = stack.enter_context(tarfile.open(file))
            except tarfile.ReadError:
                logger.debug(
                    f"Failed to find main tex file of {self.arxiv_id}: Not a tar file."
                )
                return None

            tex_files = [f for f in tar.getnames() if f.endswith(".tex")]
            if len(tex_files) == 0:
                logger.debug(
                    f"Failed to find main tex file of {self.arxiv_id}: No tex file."
                )
                return None

            bbl_file = [f for f in tar.getnames() if f.endswith(".bbl")]
            match len(bbl_file):
                case 0:
                    if len(tex_files) > 1:
                        logger.debug(
                            f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple tex files while no bbl file."
                        )
                        main_tex = None
                    else:
                        main_tex = tex_files[0]
                case 1:
                    main_name = bbl_file[0].replace(".bbl", "")
                    main_tex = f"{main_name}.tex"
                    if main_tex not in tex_files:
                        logger.debug(
                            f"Cannot find main tex file of {self.arxiv_id} from bbl: The bbl file does not match any tex file."
                        )
                        main_tex = None
                case _:
                    logger.debug(
                        f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple bbl files."
                    )
                    main_tex = None
            if main_tex is None:
                logger.debug(
                    f"Trying to choose tex file containing the document block as main tex file of {self.arxiv_id}"
                )

            file_contents = {}
            for t in tex_files:
                f = tar.extractfile(t)
                content = f.read().decode("utf-8", errors="ignore")
                content = re.sub(r"%.*\n", "\n", content)
                content = re.sub(
                    r"\\begin{comment}.*?\\end{comment}",
                    "",
                    content,
                    flags=re.DOTALL,
                )
                content = re.sub(r"\\iffalse.*?\\fi", "", content, flags=re.DOTALL)
                content = re.sub(r"\n+", "\n", content)
                content = re.sub(r"\\\\", "", content)
                content = re.sub(r"[ \t\r\f]{3,}", " ", content)
                if main_tex is None and re.search(r"\\begin\{document\}", content):
                    main_tex = t
                    logger.debug(f"Choose {t} as main tex file of {self.arxiv_id}")
                file_contents[t] = content

            if main_tex is not None:
                main_source = file_contents[main_tex]
                include_files = re.findall(
                    r"\\input\{(.+?)\}", main_source
                ) + re.findall(r"\\include\{(.+?)\}", main_source)
                for f in include_files:
                    file_name = f if f.endswith(".tex") else f"{f}.tex"
                    main_source = main_source.replace(
                        f"\\input{{{f}}}", file_contents.get(file_name, "")
                    )
                file_contents["all"] = main_source
            else:
                logger.debug(
                    f"Failed to find main tex file of {self.arxiv_id}: No tex file containing the document block."
                )
                file_contents["all"] = None
        return file_contents

    def _build_tldr_prompt(self, lang: str) -> str:
        introduction = ""
        conclusion = ""
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(self.tex.values())
            content = re.sub(r"~?\\cite.?\{.*?\}", "", content)
            content = re.sub(
                r"\\begin\{figure\}.*?\\end\{figure\}",
                "",
                content,
                flags=re.DOTALL,
            )
            content = re.sub(
                r"\\begin\{table\}.*?\\end\{table\}",
                "",
                content,
                flags=re.DOTALL,
            )
            match = re.search(
                r"\\section\{Introduction\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)",
                content,
                flags=re.DOTALL,
            )
            if match:
                introduction = match.group(0)
            match = re.search(
                r"\\section\{Conclusion\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)",
                content,
                flags=re.DOTALL,
            )
            if match:
                conclusion = match.group(0)
        prompt = """Given the title, abstract, introduction and the conclusion (if any) of a paper in latex format, generate a one-sentence TLDR summary in __LANG__:

\\title{__TITLE__}
\\begin{abstract}__ABSTRACT__\\end{abstract}
__INTRODUCTION__
__CONCLUSION__
"""
        prompt = prompt.replace("__LANG__", lang)
        prompt = prompt.replace("__TITLE__", self.title)
        prompt = prompt.replace("__ABSTRACT__", self.summary)
        prompt = prompt.replace("__INTRODUCTION__", introduction)
        prompt = prompt.replace("__CONCLUSION__", conclusion)
        return prompt

    @cached_property
    def tldr_en(self) -> str:
        return _generate_scientific_tldr(self._build_tldr_prompt("English"))

    @cached_property
    def tldr_zh(self) -> str:
        return _translate_tldr(self.tldr_en)

    @cached_property
    def tldr(self) -> str:
        return self.tldr_en

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        if self.tex is None:
            return None
        content = self.tex.get("all")
        if content is None:
            content = "\n".join(self.tex.values())
        possible_regions = [r"\\author.*?\\maketitle", r"\\begin{document}.*?\\begin{abstract}"]
        matches = [re.search(p, content, flags=re.DOTALL) for p in possible_regions]
        match = next((m for m in matches if m), None)
        if match:
            information_region = match.group(0)
        else:
            logger.debug(
                f"Failed to extract affiliations of {self.arxiv_id}: No author information found."
            )
            return None
        prompt = (
            "Given the author information of a paper in latex format, extract the affiliations "
            "of the authors in a python list format, which is sorted by the author order. "
            "If there is no affiliation found, return an empty list '[]'. Following is the "
            f"author information:\n{information_region}"
        )
        llm = get_llm()
        affiliations = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who perfectly extracts affiliations of authors from the author information of a paper. You should return a python list of affiliations sorted by the author order, like ['TsingHua University','Peking University']. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                },
                {"role": "user", "content": _truncate_prompt(prompt)},
            ]
        )
        try:
            affiliations = re.search(r"\[.*?\]", affiliations, flags=re.DOTALL).group(0)
            affiliations = eval(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]
        except Exception as e:
            logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: {e}")
            return None
        return affiliations


class BiorxivPaper:
    def __init__(self, paper: dict):
        self._paper = paper
        self.score = None

    @property
    def title(self) -> str:
        return self._paper["title"]

    @cached_property
    def summary(self) -> str:
        return self._paper["abstract"]

    @property
    def authors(self) -> list[str]:
        return self._paper["authors"].split(";")

    @property
    def biorxiv_id(self) -> str:
        return self._paper["doi"]

    @property
    def paper_url(self) -> str:
        return f"https://www.biorxiv.org/content/{self.biorxiv_id}v{self._paper['version']}"

    @property
    def code_url(self) -> Optional[str]:
        return None

    @property
    def category(self) -> str:
        return self._paper["category"]

    @property
    def institution(self) -> str:
        return self._paper["author_corresponding_institution"]

    @property
    def update_time(self) -> str:
        return self._paper["date"]

    def _build_tldr_prompt(self, lang: str) -> str:
        prompt = """Given the title and abstract of a paper in latex format, generate a one-sentence TLDR summary in __LANG__:

\\title{__TITLE__}
\\begin{abstract}__ABSTRACT__\\end{abstract}
"""
        prompt = prompt.replace("__LANG__", lang)
        prompt = prompt.replace("__TITLE__", self.title)
        prompt = prompt.replace("__ABSTRACT__", self.summary)
        return prompt

    @cached_property
    def tldr_en(self) -> str:
        return _generate_scientific_tldr(self._build_tldr_prompt("English"))

    @cached_property
    def tldr_zh(self) -> str:
        return _translate_tldr(self.tldr_en)

    @cached_property
    def tldr(self) -> str:
        return self.tldr_en


class JournalPaper:
    def __init__(self, paper: dict):
        self._paper = paper
        self.score = None

    @property
    def title(self) -> str:
        return self._paper["title"]

    @cached_property
    def summary(self) -> str:
        return self._paper["abstract"]

    @property
    def authors(self) -> list[str]:
        return self._paper["authors"]

    @property
    def paper_id(self) -> str:
        return self._paper["paper_id"]

    @property
    def paper_url(self) -> str:
        return self._paper["paper_url"]

    @property
    def code_url(self) -> Optional[str]:
        return None

    @property
    def journal(self) -> str:
        return self._paper["journal"]

    @property
    def published_at(self) -> str:
        return self._paper["published_at"]

    def _build_tldr_prompt(self, lang: str) -> str:
        prompt = """Given the title and abstract of a paper in latex format, generate a one-sentence TLDR summary in __LANG__:

\\title{__TITLE__}
\\begin{abstract}__ABSTRACT__\\end{abstract}
"""
        prompt = prompt.replace("__LANG__", lang)
        prompt = prompt.replace("__TITLE__", self.title)
        prompt = prompt.replace("__ABSTRACT__", self.summary)
        return prompt

    @cached_property
    def tldr_en(self) -> str:
        return _generate_scientific_tldr(self._build_tldr_prompt("English"))

    @cached_property
    def tldr_zh(self) -> str:
        return _translate_tldr(self.tldr_en)

    @cached_property
    def tldr(self) -> str:
        return self.tldr_en
