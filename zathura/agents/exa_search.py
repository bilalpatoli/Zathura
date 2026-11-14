import os

from exa_py import Exa

from zathura.schema import Source


def search_papers(query, n=3):
    if not isinstance(n, int) or not 1 <= n <= 20:
        raise ValueError("n must be an integer between 1 and 20")
    if not query.strip():
        raise ValueError("query must not be empty")
    key = os.getenv("EXA_API_KEY")
    if not key or os.getenv("ZATHURA_MOCK") == "1":
        sources = [
            Source(title="[MOCK] Box-fitting algorithms for periodic transits (Kovacs et al.)",
                   url="https://arxiv.org/abs/astro-ph/0206099",
                   snippet="MOCK canned summary: BLS searches periodic box-shaped flux decreases; grid spacing affects recovery.", mock=True),
            Source(title="[MOCK] Transit Least Squares (Hippke & Heller)",
                   url="https://arxiv.org/abs/1901.02015",
                   snippet="MOCK canned summary: shallow-transit sensitivity depends on detrending and duration coverage.", mock=True),
            Source(title="[MOCK] Kepler data processing handbook",
                   url="https://archive.stsci.edu/kepler/manuals/",
                   snippet="MOCK canned summary: stellar variability and instrumental systematics affect transit searches.", mock=True),
        ]
        return [s.model_dump() for s in sources[:n]]
    response = Exa(api_key=key).search_and_contents(query, num_results=n, text={"max_characters": 1500})
    return [Source(title=r.title or "Untitled", url=r.url, snippet=(r.text or "")[:1500]).model_dump()
            for r in response.results]
