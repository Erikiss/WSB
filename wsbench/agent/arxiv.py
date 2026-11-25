"""Fetch an arXiv paper for the agent: metadata and abstract from the arXiv API, full text from arXiv's HTML version.

Only arXiv is contacted, and only for a well-formed arXiv identifier. Texts are cached in WSB_HOME/papers/.
"""
import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import httpx

from wsbench.paths import home

USER_AGENT = "wsbench/0.1 (https://github.com/Erikiss/WSB)"
ARXIV_ID = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv:)?(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
PART_CHARS = 15_000
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def parse_id(reference):
    match = ARXIV_ID.search(str(reference).strip())
    if match is None:
        raise ValueError("not an arXiv identifier; expected something like 2611.30835 or https://arxiv.org/abs/2611.30835")
    return match.group(1)


class _Text(HTMLParser):
    """Visible text of an arXiv HTML paper: skips scripts, styles, navigation and the MathML duplicates of formulas."""
    SKIP = {"script", "style", "nav", "header", "footer", "annotation-xml", "button", "svg"}
    BLOCK = {"p", "div", "section", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "figcaption", "br", "table"}

    def __init__(self):
        super().__init__()
        self.out, self.skip, self.in_math, self.headings, self.heading = [], 0, 0, [], None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "math":
            self.in_math += 1
            alt = dict(attrs).get("alttext")
            if alt and not self.skip:
                self.out.append(f" ${alt}$ ")
        elif tag in self.BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3") and not self.skip:
            self.heading = []

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "math":
            self.in_math = max(0, self.in_math - 1)
        elif tag in ("h1", "h2", "h3") and self.heading is not None:
            title = " ".join("".join(self.heading).split())
            if title:
                self.headings.append(title)
            self.heading = None
            self.out.append("\n")

    def handle_data(self, data):
        if self.skip or self.in_math:
            return
        self.out.append(data)
        if self.heading is not None:
            self.heading.append(data)

    def text(self):
        lines = (" ".join(line.split()) for line in "".join(self.out).splitlines())
        return "\n".join(line for line in lines if line)


def _fetch(arxiv_id, get):
    meta = get(f"https://export.arxiv.org/api/query?id_list={arxiv_id}")
    entry = ET.fromstring(meta).find("a:entry", ATOM)
    if entry is None or entry.find("a:title", ATOM) is None:
        raise ValueError(f"arXiv has no paper {arxiv_id}")
    clean = lambda node: " ".join((node.text or "").split()) if node is not None else ""
    paper = dict(id=arxiv_id, title=clean(entry.find("a:title", ATOM)), abstract=clean(entry.find("a:summary", ATOM)),
                 authors=[clean(a.find("a:name", ATOM)) for a in entry.findall("a:author", ATOM)][:12],
                 published=clean(entry.find("a:published", ATOM))[:10], text="", headings=[])
    try:
        parser = _Text()
        parser.feed(get(f"https://arxiv.org/html/{arxiv_id}"))
        paper.update(text=parser.text(), headings=parser.headings[:60])
    except Exception:
        pass                                  # older papers have no HTML version: the abstract is all we have
    return paper


def _http_get(url):
    response = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=40.0, follow_redirects=True)
    response.raise_for_status()
    return response.text


def load(reference, get=_http_get):
    arxiv_id = parse_id(reference)
    path = home() / "papers" / f"{arxiv_id}.json"
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        pass
    paper = _fetch(arxiv_id, get)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(paper), "utf-8")
    except OSError:
        pass
    return paper


def read(reference, part=0, get=_http_get):
    """part 0: title, authors, abstract, section headings and the number of text parts. part n >= 1: text part n."""
    paper = load(reference, get)
    parts = -(-len(paper["text"]) // PART_CHARS)
    if part == 0:
        return dict(id=paper["id"], title=paper["title"], authors=paper["authors"], published=paper["published"],
                    abstract=paper["abstract"], headings=paper["headings"], text_parts=parts,
                    note="Call again with part=1..text_parts to read the full text." if parts
                    else "No HTML full text is available for this paper; only the abstract.")
    if not 1 <= part <= parts:
        raise ValueError(f"part must be between 0 and {parts}")
    return dict(id=paper["id"], part=part, text_parts=parts, text=paper["text"][(part - 1) * PART_CHARS: part * PART_CHARS])
