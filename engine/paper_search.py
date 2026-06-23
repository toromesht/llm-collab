#!/usr/bin/env python3
"""
paper_search.py — AI Paper Search API Integration

Sources:
  Semantic Scholar  — 214M+ papers, TLDR summaries, citation graphs (free, API key optional)
  arXiv             — 2.4M+ CS/ML preprints, true LaTeX search (free, no key)
  OpenAlex          — 250M+ works, concept tagging (free, polite pool)

Usage:
  from engine.paper_search import search_papers, get_paper, find_similar
  papers = search_papers("grid cells", source="semantic_scholar", limit=10)
  paper  = get_paper("ARXIV:1506.02640")
  recs   = find_similar("10.1038/nature03721")

Quick start:
  python engine/paper_search.py "transformer attention"  # search all sources
  python engine/paper_search.py --source arxiv "grid cell" --limit 20
"""

import json
import time
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# 1. SEMANTIC SCHOLAR API
# ═══════════════════════════════════════════════════════════════
# Free API key: https://www.semanticscholar.org/product/api#api-key-form
# Without key: shared 1000 req/s pool (may throttle)
# With key: dedicated 1 req/s quota (can request increase)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = None  # Set to your key from semanticscholar.org

_s2_headers = {}
if S2_API_KEY:
    _s2_headers["x-api-key"] = S2_API_KEY


def search_semantic_scholar(query: str, limit: int = 10,
                            year: str = None, fields: str = None) -> list:
    """
    Search papers on Semantic Scholar.

    Args:
        query: Search query (supports +required -excluded "phrase" ~N proximity)
        limit: Max results (default 10, max 100)
        year: Year filter e.g. "2024-" for 2024+, "2020-2025" for range
        fields: Comma-separated fields. Default: title,authors,year,citationCount,abstract,tldr

    Returns:
        List of paper dicts with keys: paperId, title, year, authors, abstract, tldr, url, ...
    """
    if fields is None:
        fields = "title,authors,year,citationCount,abstract,tldr,url,externalIds,publicationVenue"

    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": fields,
    }
    if year:
        params["year"] = year

    url = f"{S2_BASE}/paper/search/bulk?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url, headers=_s2_headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return [{"error": f"HTTP {e.code}", "detail": body[:200]}]
    except Exception as e:
        return [{"error": str(e)}]

    papers = data.get("data", [])
    # Clean up: extract first author, format venue
    for p in papers:
        authors = p.get("authors", [])
        p["first_author"] = authors[0].get("name", "?") if authors else "?"
        p["author_count"] = len(authors)
        p["author_names"] = [a.get("name", "?") for a in authors[:5]]
        # TLDR: AI-generated one-sentence summary
        p["tldr_text"] = p.get("tldr", {}).get("text", "") if p.get("tldr") else ""
        # DOI / ArXiv ID
        ext = p.get("externalIds", {})
        p["doi"] = ext.get("DOI", "")
        p["arxiv_id"] = ext.get("ArXiv", "")
        if p.get("url"):
            p["semantic_url"] = p["url"]
    return papers


def get_paper_s2(paper_id: str) -> dict:
    """Get full details for a paper by ID. Accepts DOI:, ARXIV:, PMID: prefixes."""
    fields = ("title,authors,year,citationCount,abstract,tldr,url,"
              "externalIds,publicationVenue,referenceCount,citations")
    url = f"{S2_BASE}/paper/{urllib.parse.quote(paper_id)}?fields={fields}"
    try:
        req = urllib.request.Request(url, headers=_s2_headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def find_similar(paper_id: str, limit: int = 10) -> list:
    """Get paper recommendations similar to a given paper."""
    url = (f"{S2_BASE}/recommendations/v1/papers/"
           f"forpaper/{urllib.parse.quote(paper_id)}"
           f"?limit={min(limit, 500)}&fields=title,authors,year,citationCount")
    try:
        req = urllib.request.Request(url, headers=_s2_headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        return data.get("recommendedPapers", [])
    except Exception as e:
        return [{"error": str(e)}]


# ═══════════════════════════════════════════════════════════════
# 2. ARXIV API
# ═══════════════════════════════════════════════════════════════
# Rate limit: 1 request per 3 seconds (be polite)
# Categories: cs.AI, cs.LG, cs.CL, cs.CV, stat.ML, q-bio.NC, etc.

ARXIV_BASE = "http://export.arxiv.org/api/query"

# AI/ML relevant categories
ARXIV_AI_CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE",
    "stat.ML", "q-bio.NC", "q-bio.QM", "physics.med-ph",
]


def search_arxiv(query: str, limit: int = 10,
                 categories: list = None, sort_by: str = "relevance") -> list:
    """
    Search arXiv for preprints.

    Args:
        query: Search terms. Supports field prefixes:
               ti:title, au:author, abs:abstract, cat:category
               e.g. "ti:grid cell AND cat:q-bio.NC"
        limit: Max results (max 2000)
        categories: List of arXiv categories e.g. ["cs.AI", "cs.LG"]
        sort_by: "relevance" | "lastUpdatedDate" | "submittedDate"

    Returns:
        List of paper dicts
    """
    # Build query with category filters
    full_query = query
    if categories:
        cat_filter = " OR ".join(f"cat:{c}" for c in categories)
        full_query = f"({query}) AND ({cat_filter})"

    params = {
        "search_query": full_query,
        "start": 0,
        "max_results": min(limit, 100),
        "sortBy": sort_by,
    }
    url = ARXIV_BASE + "?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_data = resp.read().decode()
    except Exception as e:
        return [{"error": str(e)}]

    # Parse Atom XML
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)

        authors = []
        for author in entry.findall("atom:author", ns):
            name_el = author.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        links = []
        for link in entry.findall("atom:link", ns):
            href = link.get("href", "")
            title = link.get("title", "")
            links.append({"href": href, "title": title})

        # Extract arXiv ID from the ID URL
        id_url = entry.find("atom:id", ns)
        arxiv_id = ""
        if id_url is not None and id_url.text:
            arxiv_id = id_url.text.split("/abs/")[-1]

        # Categories
        cats = []
        for cat in entry.findall("arxiv:primary_category", ns):
            cats.append(cat.get("term", ""))
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term", "")
            if term not in cats:
                cats.append(term)

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""

        papers.append({
            "paperId": f"ARXIV:{arxiv_id}",
            "title": title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else "",
            "authors": authors,
            "first_author": authors[0] if authors else "?",
            "author_count": len(authors),
            "year": published_el.text[:4] if published_el is not None and published_el.text else "",
            "abstract": summary_el.text.strip().replace("\n", " ") if summary_el is not None and summary_el.text else "",
            "categories": cats,
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "source": "arxiv",
        })
    return papers


# ═══════════════════════════════════════════════════════════════
# 3. UNIFIED SEARCH
# ═══════════════════════════════════════════════════════════════

def search_papers(query: str, source: str = "all", limit: int = 10,
                  year: str = None, categories: list = None) -> dict:
    """
    Unified search across all sources.

    Args:
        query: Search terms
        source: "semantic_scholar" | "arxiv" | "all" (default: all)
        limit: Max results per source
        year: Year filter (Semantic Scholar only)
        categories: arXiv categories filter

    Returns:
        {"query": str, "sources": {"semantic_scholar": [...], "arxiv": [...]}}
    """
    results = {"query": query, "sources": {}}

    if source in ("semantic_scholar", "all"):
        s2_papers = search_semantic_scholar(query, limit=limit, year=year)
        results["sources"]["semantic_scholar"] = s2_papers

    if source in ("arxiv", "all"):
        arxiv_papers = search_arxiv(query, limit=limit, categories=categories)
        results["sources"]["arxiv"] = arxiv_papers

    return results


def get_paper(paper_id: str) -> dict:
    """Get paper details. Accepts DOI:10.xxx, ARXIV:xxxx.xxxxx, or raw paperId."""
    if paper_id.upper().startswith("ARXIV:"):
        # Search arXiv by ID
        arxiv_id = paper_id.split(":", 1)[1]
        papers = search_arxiv(f"id:{arxiv_id}", limit=1)
        return papers[0] if papers else {"error": "not found"}
    else:
        return get_paper_s2(paper_id)


# ═══════════════════════════════════════════════════════════════
# MAIN — CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("\nExamples:")
        print("  python paper_search.py 'transformer attention mechanism'")
        print("  python paper_search.py --source arxiv 'grid cell' --limit 5")
        print("  python paper_search.py --source s2 'large language model routing' --year 2024-")
        print("  python paper_search.py --get ARXIV:1506.02640")
        print("  python paper_search.py --similar DOI:10.1038/nature03721")
        sys.exit(0)

    if "--get" in sys.argv:
        idx = sys.argv.index("--get")
        paper_id = sys.argv[idx + 1]
        paper = get_paper(paper_id)
        print(json.dumps(paper, indent=2, ensure_ascii=False))
        sys.exit(0)

    if "--similar" in sys.argv:
        idx = sys.argv.index("--similar")
        paper_id = sys.argv[idx + 1]
        recs = find_similar(paper_id, limit=10)
        for i, r in enumerate(recs):
            print(f"{i+1}. {r.get('title','?')} ({r.get('year','?')}) — cited {r.get('citationCount',0)}x")
        sys.exit(0)

    source = "all"
    limit = 10
    year = None
    query_parts = []

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--source" and i + 1 < len(sys.argv):
            source = sys.argv[i + 1]; i += 2
        elif arg == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1]); i += 2
        elif arg == "--year" and i + 1 < len(sys.argv):
            year = sys.argv[i + 1]; i += 2
        else:
            query_parts.append(arg); i += 1

    query = " ".join(query_parts)
    if not query:
        print("Error: no query provided"); sys.exit(1)

    results = search_papers(query, source=source, limit=limit, year=year)

    for src_name, papers in results["sources"].items():
        print(f"\n{'='*60}")
        print(f"  {src_name.upper()} — {len(papers)} results for: {query}")
        print(f"{'='*60}")
        for j, p in enumerate(papers):
            if "error" in p:
                print(f"  ERROR: {p['error']}")
                continue
            title = p.get("title", "?")[:100]
            author = p.get("first_author", "?")
            yr = p.get("year", "?")
            citations = p.get("citationCount", "")
            tldr = p.get("tldr_text", "")
            arxiv = p.get("arxiv_id", "")
            doi = p.get("doi", "")
            print(f"\n{j+1}. {title}")
            print(f"   {author} et al. ({yr})" +
                  (f" — cited {citations}x" if citations else ""))
            if tldr:
                print(f"   TLDR: {tldr}")
            ids = []
            if arxiv: ids.append(f"arXiv:{arxiv}")
            if doi: ids.append(f"DOI:{doi}")
            if ids: print(f"   {' | '.join(ids)}")
