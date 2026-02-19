# search/retrieve.py
from typing import Dict, Any, List
from search.hybrid_search import hybrid_search

def retrieve(query: str, k: int = 8) -> Dict[str, Any]:
    hits = hybrid_search(query, top_k=k)

    context_blocks: List[str] = []
    citations: List[Dict[str, Any]] = []

    for i, h in enumerate(hits, start=1):
        loc = f"{h.get('doc_title')} {h.get('unit_id') or ''} {h.get('path') or ''}".strip()
        text = h.get("text", "")

        context_blocks.append(
            f"[근거{i}] {loc}\n{text}"
        )
        citations.append({
            "rank": i,
            "doc_title": h.get("doc_title"),
            "unit_id": h.get("unit_id"),
            "path": h.get("path"),
            "doc_id": h.get("doc_id"),
            "score": h.get("score"),
        })

    return {
        "query": query,
        "hits": hits,
        "context": "\n\n".join(context_blocks),
        "citations": citations,
    }
