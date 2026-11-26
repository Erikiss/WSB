"""Fetch a LessWrong or Alignment Forum post for the agent through the site's public GraphQL API.

Only the forum's own API is contacted, and only for a well-formed post identifier. Texts are cached in WSB_HOME/posts/.
"""
import json
import re

import httpx

from wsbench.paths import home

from .arxiv import PART_CHARS, USER_AGENT, _Text

POST_URL = re.compile(r"(?:lesswrong\.com|alignmentforum\.org)/posts/([A-Za-z0-9]{17})(?:[/?#]|$)", re.IGNORECASE)
BARE_ID = re.compile(r"[A-Za-z0-9]{17}")
API = "https://www.lesswrong.com/graphql"
QUERY = "query($id: String){ post(input:{selector:{_id:$id}}){ result{ title postedAt baseScore user{displayName} htmlBody } } }"


def parse_id(reference):
    reference = str(reference).strip()
    match = POST_URL.search(reference)
    if match:
        return match.group(1)
    if BARE_ID.fullmatch(reference):
        return reference
    raise ValueError("not a LessWrong post; expected a link like https://www.lesswrong.com/posts/<id>/<title>")


def _http_post(post_id):
    response = httpx.post(API, json={"query": QUERY, "variables": {"id": post_id}},
                          headers={"User-Agent": USER_AGENT}, timeout=40.0)
    response.raise_for_status()
    return response.json()


def load(reference, fetch=_http_post):
    post_id = parse_id(reference)
    path = home() / "posts" / f"{post_id}.json"
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        pass
    result = (((fetch(post_id).get("data") or {}).get("post") or {}).get("result"))
    if not result:
        raise ValueError(f"LessWrong has no post {post_id}")
    parser = _Text()
    parser.feed(result.get("htmlBody") or "")
    post = dict(id=post_id, title=result.get("title") or "", author=(result.get("user") or {}).get("displayName") or "",
                posted=(result.get("postedAt") or "")[:10], karma=result.get("baseScore"), text=parser.text(),
                headings=parser.headings[:60])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(post), "utf-8")
    except OSError:
        pass
    return post


def read(reference, part=0, fetch=_http_post):
    """part 0: title, author, date, karma, headings and the number of text parts. part n >= 1: text part n."""
    post = load(reference, fetch)
    parts = -(-len(post["text"]) // PART_CHARS)
    if part == 0:
        return dict(id=post["id"], title=post["title"], author=post["author"], posted=post["posted"], karma=post["karma"],
                    headings=post["headings"], text_parts=parts, opening=post["text"][:1500],
                    note="Call again with part=1..text_parts to read the full text.")
    if not 1 <= part <= parts:
        raise ValueError(f"part must be between 0 and {parts}")
    return dict(id=post["id"], part=part, text_parts=parts, text=post["text"][(part - 1) * PART_CHARS: part * PART_CHARS])
