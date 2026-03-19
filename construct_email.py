import datetime
import math
import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr

from tqdm import tqdm

from paper import ArxivPaper, BiorxivPaper, JournalPaper

framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em;
      line-height: 1;
      display: inline-flex;
      align-items: center;
    }
    .half-star {
      display: inline-block;
      width: 0.5em;
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<h1>Arxiv Papers</h1>
<div>
    __CONTENT-ARXIV__
</div>

<h1>BioRxiv Papers</h1>
<div>
    __CONTENT-BIORXIV__
</div>

<h1>Journal Papers</h1>
<div>
    __CONTENT-JOURNAL__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""


def get_empty_html():
    return """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No new papers matched your query today.
    </td>
  </tr>
  </table>
  """


def get_block_html(
    title: str,
    authors: str,
    submeta: str,
    rate: str,
    identifier_label: str,
    identifier_value: str,
    identifier_url: str,
    tldr_en: str,
    tldr_zh: str,
    primary_link_url: str,
    primary_link_label: str,
    secondary_link_url: str = None,
    secondary_link_label: str = None,
):
    secondary_link = (
        f'<a href="{secondary_link_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">{secondary_link_label}</a>'
        if secondary_link_url and secondary_link_label
        else ""
    )
    return f"""
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{submeta}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>{identifier_label}:</strong> <a href="{identifier_url}" target="_blank">{identifier_value}</a>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>English TLDR:</strong> {tldr_en}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>中文翻译:</strong> {tldr_zh}
        </td>
    </tr>
    <tr>
        <td style="padding: 8px 0;">
            <a href="{primary_link_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">{primary_link_label}</a>
            {secondary_link}
        </td>
    </tr>
</table>
"""


def get_stars(score: float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score is None or score <= low:
        return ""
    if score >= high:
        return full_star * 5
    interval = (high - low) / 10
    star_num = math.ceil((score - low) / interval)
    full_star_num = int(star_num / 2)
    half_star_num = star_num - full_star_num * 2
    return (
        '<div class="star-wrapper">'
        + full_star * full_star_num
        + half_star * half_star_num
        + "</div>"
    )


def _join_authors(author_list: list[str]) -> str:
    if len(author_list) <= 5:
        return ", ".join(author_list)
    return ", ".join(author_list[:3] + ["..."] + author_list[-2:])


def _format_arxiv_block(paper: ArxivPaper) -> str:
    authors = _join_authors([author.name for author in paper.authors])
    if paper.affiliations is not None:
        affiliations = ", ".join(paper.affiliations[:5])
        if len(paper.affiliations) > 5:
            affiliations += ", ..."
    else:
        affiliations = "Unknown Affiliation"
    return get_block_html(
        title=paper.title,
        authors=authors,
        submeta=affiliations,
        rate=get_stars(paper.score),
        identifier_label="arXiv ID",
        identifier_value=paper.arxiv_id,
        identifier_url=f"https://arxiv.org/abs/{paper.arxiv_id}",
        tldr_en=paper.tldr_en,
        tldr_zh=paper.tldr_zh,
        primary_link_url=paper.pdf_url,
        primary_link_label="PDF",
        secondary_link_url=paper.code_url,
        secondary_link_label="Code",
    )


def _format_biorxiv_block(paper: BiorxivPaper) -> str:
    authors = _join_authors(paper.authors)
    affiliations = paper.institution or "Unknown Affiliation"
    return get_block_html(
        title=paper.title,
        authors=authors,
        submeta=affiliations,
        rate=get_stars(paper.score),
        identifier_label="bioRxiv DOI",
        identifier_value=paper.biorxiv_id,
        identifier_url=f"https://doi.org/{paper.biorxiv_id}",
        tldr_en=paper.tldr_en,
        tldr_zh=paper.tldr_zh,
        primary_link_url=paper.paper_url,
        primary_link_label="Paper",
        secondary_link_url=paper.code_url,
        secondary_link_label="Code",
    )


def _format_journal_block(paper: JournalPaper) -> str:
    authors = _join_authors(paper.authors)
    submeta = paper.journal
    if paper.published_at:
        submeta = f"{submeta} | {paper.published_at}"
    identifier_label = "DOI" if "/" in paper.paper_id else "PMID"
    identifier_url = (
        f"https://doi.org/{paper.paper_id}"
        if identifier_label == "DOI"
        else f"https://pubmed.ncbi.nlm.nih.gov/{paper.paper_id}/"
    )
    return get_block_html(
        title=paper.title,
        authors=authors,
        submeta=submeta,
        rate=get_stars(paper.score),
        identifier_label=identifier_label,
        identifier_value=paper.paper_id,
        identifier_url=identifier_url,
        tldr_en=paper.tldr_en,
        tldr_zh=paper.tldr_zh,
        primary_link_url=paper.paper_url,
        primary_link_label="Article",
    )


def _render_section(papers, formatter, desc: str) -> str:
    if len(papers) == 0:
        return get_empty_html()
    parts = [formatter(paper) for paper in tqdm(papers, desc=desc)]
    return "<br>" + "</br><br>".join(parts) + "</br>"


def render_email(
    papers: list[ArxivPaper],
    papers_biorxiv: list[BiorxivPaper],
    papers_journal: list[JournalPaper],
):
    html = framework.replace(
        "__CONTENT-ARXIV__", _render_section(papers, _format_arxiv_block, "Rendering arXiv email")
    )
    html = html.replace(
        "__CONTENT-BIORXIV__",
        _render_section(papers_biorxiv, _format_biorxiv_block, "Rendering bioRxiv email"),
    )
    html = html.replace(
        "__CONTENT-JOURNAL__",
        _render_section(papers_journal, _format_journal_block, "Rendering journal email"),
    )
    return html


def send_email(sender: str, receiver: str, password: str, smtp_server: str, smtp_port: int, html: str):
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, "utf-8").encode(), addr))

    msg = MIMEText(html, "html", "utf-8")
    msg["From"] = _format_addr("Github Action <%s>" % sender)
    msg["To"] = _format_addr("You <%s>" % receiver)
    today = datetime.datetime.now().strftime("%Y/%m/%d")
    msg["Subject"] = Header(f"Daily Papers {today}", "utf-8").encode()

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()
