import sys
from pathlib import Path

# --- sys.path 보정: 직접 실행/모듈 실행 모두 안정화
BASE_DIR = Path(__file__).resolve().parents[1]  # .../law_ch
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import pickle
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from d_p.ju_to_em import embed_query  # ✅ 질문 임베딩은 이 함수로 고정


# =========================
# 파일 경로
# =========================
EMBED_DIR = BASE_DIR / "embedded"
DOCS_PATH = EMBED_DIR / "docs.jsonl"
FAISS_PATH = EMBED_DIR / "faiss.index"
BM25_PATH = EMBED_DIR / "bm25.pkl"


# =========================
# BM25 토큰화 (build_bm25.py와 동일해야 함)
# =========================
def tokenize_kr_mvp(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text.split(" ") if text else []


# =========================
# 질문 임베딩 (KURE-v1 via ju_to_em.embed_query)
# =========================
def embed_query_kure(query: str) -> np.ndarray:
    """
    ju_to_em.embed_query()는 이미 np.float32 1D를 반환하도록 구현되어 있음.
    (detach/cpu 처리 불필요)
    """
    v = embed_query(query)
    return np.asarray(v, dtype=np.float32).reshape(-1)


def rank_to_score(indices: List[int]) -> Dict[int, float]:
    n = len(indices)
    if n == 0:
        return {}
    denom = max(1, n - 1)
    return {idx: 1.0 - (r / denom) for r, idx in enumerate(indices)}


# =========================
# 리소스(문서/인덱스/BM25) 1회 로드 & 재사용
# =========================
_DOCS: Optional[List[Dict[str, Any]]] = None
_BM25 = None
_FAISS_INDEX = None
_FAISS = None  # faiss 모듈 (지연 import)


def load_docs_once() -> List[Dict[str, Any]]:
    global _DOCS
    if _DOCS is not None:
        return _DOCS

    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"docs.jsonl not found: {DOCS_PATH}")

    docs: List[Dict[str, Any]] = []
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))

    _DOCS = docs
    return _DOCS


def load_bm25_once():
    global _BM25
    if _BM25 is not None:
        return _BM25

    if not BM25_PATH.exists():
        raise FileNotFoundError(f"bm25 not found: {BM25_PATH}")

    with BM25_PATH.open("rb") as f:
        _BM25 = pickle.load(f)
    return _BM25


def load_faiss_once():
    global _FAISS_INDEX, _FAISS
    if _FAISS_INDEX is not None:
        return _FAISS_INDEX

    if not FAISS_PATH.exists():
        raise FileNotFoundError(f"faiss index not found: {FAISS_PATH}")

    import faiss  # 지연 import
    _FAISS = faiss
    _FAISS_INDEX = faiss.read_index(str(FAISS_PATH))
    return _FAISS_INDEX


def warmup():
    """
    프로그램 시작 시 1회 로딩:
    - docs / bm25 / faiss index
    - 그리고 첫 embed_query() 호출로 모델도 미리 올릴 수 있음(선택)
    """
    load_docs_once()
    load_bm25_once()
    load_faiss_once()

    # (선택) 모델까지 미리 올려서 첫 질문 지연을 없앰
    # 너무 무겁다면 주석 처리해도 됨.
    _ = embed_query_kure("워밍업")


# =========================
# 하이브리드 검색 (로딩 없음, 순수 검색만)
# =========================
def hybrid_search(
    query: str,
    top_k: int = 10,
    topN_vec: int = 50,
    topN_bm25: int = 50,
    w_vec: float = 0.6,
    w_bm25: float = 0.4,
) -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []

    docs = load_docs_once()
    index = load_faiss_once()
    bm25 = load_bm25_once()
    faiss = _FAISS  # load_faiss_once()에서 세팅됨

    # (A) Vector retrieve
    qv = embed_query_kure(query).reshape(1, -1).astype(np.float32)
    # 코사인 유사도(IP + normalize) 기반이면 정규화 필요
    faiss.normalize_L2(qv)

    _, vec_idxs = index.search(qv, topN_vec)
    vec_list = vec_idxs[0].tolist()

    # (B) BM25 retrieve
    q_tokens = tokenize_kr_mvp(query)
    bm25_scores_all = bm25.get_scores(q_tokens)
    bm25_top = np.argsort(bm25_scores_all)[::-1][:topN_bm25].tolist()

    # (C) rank 기반 점수
    vec_rank = rank_to_score(vec_list)
    bm25_rank = rank_to_score(bm25_top)

    # (D) merge
    candidates = set(vec_list) | set(bm25_top)
    merged: List[Tuple[float, int]] = []
    for i in candidates:
        score = w_vec * vec_rank.get(i, 0.0) + w_bm25 * bm25_rank.get(i, 0.0)
        merged.append((score, i))
    merged.sort(reverse=True, key=lambda x: x[0])

    # (E) 결과
    results = []
    for score, i in merged[:top_k]:
        if i < 0 or i >= len(docs):
            continue
        d = docs[i]
        md = d.get("metadata", {}) or {}
        results.append({
            "score": float(score),
            "doc_id": d.get("doc_id"),
            "doc_title": md.get("doc_title"),
            "path": md.get("path"),
            "unit_id": md.get("unit_id"),
            "text": d.get("text"),
        })
    return results


def print_results(results: List[Dict[str, Any]], max_text: int = 220):
    if not results:
        print("(no results)")
        return

    for r in results:
        print("\n--- score:", r["score"])
        print("doc_id :", r.get("doc_id"))
        print("loc    :", r.get("doc_title"), r.get("path"), r.get("unit_id"))
        text = r.get("text") or ""
        print("text   :", (text[:max_text] + "...") if len(text) > max_text else text)


if __name__ == "__main__":
    # ✅ 시작 시 1회 로딩
    warmup()
    print("✅ Hybrid Search ready. (type 'exit' to quit)")

    while True:
        q = input("\n[QUERY] ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        out = hybrid_search(q, top_k=5)
        print_results(out)