import sys
from pathlib import Path

# --- sys.path 보정: 직접 실행/모듈 실행 모두 안정화
BASE_DIR = Path(__file__).resolve().parents[1]  # .../law_ch
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import pickle
import re
from typing import List, Dict, Any, Tuple

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
# 질문 임베딩 (KURE-v1)
# =========================
def embed_query_kure(query: str) -> np.ndarray:
    v = embed_query(query)
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.array(v, dtype=np.float32).reshape(-1)


# =========================
# docs 로드
# =========================
def load_docs() -> List[Dict[str, Any]]:
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"docs.jsonl not found: {DOCS_PATH}")
    docs = []
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def rank_to_score(indices: List[int]) -> Dict[int, float]:
    n = len(indices)
    if n == 0:
        return {}
    denom = max(1, n - 1)
    return {idx: 1.0 - (r / denom) for r, idx in enumerate(indices)}


# =========================
# 하이브리드 검색
# =========================
def hybrid_search(
    query: str,
    top_k: int = 10,
    topN_vec: int = 50,
    topN_bm25: int = 50,
    w_vec: float = 0.6,
    w_bm25: float = 0.4,
) -> List[Dict[str, Any]]:
    docs = load_docs()

    if not FAISS_PATH.exists():
        raise FileNotFoundError(f"faiss index not found: {FAISS_PATH}")
    if not BM25_PATH.exists():
        raise FileNotFoundError(f"bm25 not found: {BM25_PATH}")

    import faiss
    index = faiss.read_index(str(FAISS_PATH))

    with BM25_PATH.open("rb") as f:
        bm25 = pickle.load(f)

    # (A) Vector retrieve
    qv = embed_query_kure(query).reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(qv)  # 코사인(IP+정규화) 기반이면 필수

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
        d = docs[i]
        md = d.get("metadata", {})
        results.append({
            "score": float(score),
            "doc_id": d.get("doc_id"),
            "doc_title": md.get("doc_title"),
            "path": md.get("path"),
            "unit_id": md.get("unit_id"),
            "text": d.get("text"),
        })
    return results


if __name__ == "__main__":
    q = "이 규칙의 목적은 무엇인가?"
    print("[QUERY]", q)
    out = hybrid_search(q, top_k=5)

    for r in out:
        print("\n--- score:", r["score"])
        print("doc_id :", r["doc_id"])
        print("loc    :", r["doc_title"], r["path"], r["unit_id"])
        print("text   :", (r["text"][:220] + "...") if r["text"] else "")
