"""
ju_to_em.py (스트리밍/재개 가능 버전)
- json_units/*.json 을 순회하며 embedding_units를 배치로 임베딩
- embedded/all_laws.jsonl 에 append 저장
- 이미 저장된 id는 스킵하여 재실행 시 이어하기 가능

추가:
- hybrid_search에서 사용할 질문/문장 임베딩 함수(embed_query) 제공
  (문서 unit(dict)용 build_embedding_text와 분리)

사용법:
  python ju_to_em.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np
from sentence_transformers.SentenceTransformer import SentenceTransformer


OUT_NAME = "all_laws.jsonl"
BATCH_SIZE = 32  # CPU면 8~32 사이로 조절


# =========================
# 문서(unit dict) -> 임베딩용 텍스트 구성
# =========================
def build_embedding_text(unit: Dict[str, Any]) -> str:
    meta = unit.get("meta", {}) or {}
    doc_title = (meta.get("doc_title") or "").strip()
    doc_class = (meta.get("doc_class") or "").strip()

    path = unit.get("path", []) or []
    path_str = " ".join([p for p in path if isinstance(p, str) and p.strip()]).strip()

    label_parts = []
    if doc_title:
        label_parts.append(doc_title)
    if doc_class:
        label_parts.append(f"({doc_class})")
    if path_str:
        label_parts.append(path_str)

    label = " ".join(label_parts).strip()
    body = (unit.get("text") or "").strip()

    return f"{label}\n{body}".strip() if label else body


def make_record_id(unit: Dict[str, Any]) -> str:
    meta = unit.get("meta", {}) or {}
    doc_title = (meta.get("doc_title") or "doc").strip()
    unit_id = (unit.get("unit_id") or "").strip()
    return f"{doc_title}_{unit_id}" if unit_id else doc_title


def load_done_ids(out_path: Path) -> Set[str]:
    """이미 저장된 jsonl에서 id만 읽어 재실행 시 스킵 가능하게 함"""
    done: Set[str] = set()
    if not out_path.exists():
        return done

    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("id")
                if isinstance(rid, str):
                    done.add(rid)
            except Exception:
                continue
    return done


def iter_units(units_dir: Path) -> Iterable[Dict[str, Any]]:
    for fp in sorted(units_dir.glob("*.json")):
        doc = json.loads(fp.read_text(encoding="utf-8"))
        for u in (doc.get("embedding_units") or []):
            if isinstance(u, dict) and (u.get("text") or "").strip():
                yield u


# =========================
# KURE 모델 로더 (재사용 캐시)
# =========================
_MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    SentenceTransformer("nlpai-lab/KURE-v1") 모델을 1번만 로드해서 재사용.
    - 문서 임베딩/질문 임베딩 모두 동일 모델/정규화 사용
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("nlpai-lab/KURE-v1")
    return _MODEL


# =========================
# (추가) 질문/문장 임베딩 함수: hybrid_search에서 사용
# =========================
def embed_query(text: str) -> np.ndarray:
    """
    입력: 질문/문장 문자열 1개
    출력: 정규화된 임베딩 벡터 (1D np.ndarray float32)

    주의:
    - build_embedding_text(unit) 는 dict용(문서 전처리)
    - embed_query(text) 는 str용(질문 임베딩)
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("embed_query: empty text")

    model = get_model()
    vec = model.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0]
    return np.array(vec, dtype=np.float32)


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    units_dir = base_dir / "json_units"
    out_dir = base_dir / "embedded"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_NAME

    if not units_dir.exists():
        print(f"오류: json_units 폴더가 없습니다: {units_dir}")
        return 2

    done_ids = load_done_ids(out_path)
    if done_ids:
        print(f"재개 모드: 이미 처리된 레코드 {len(done_ids)}개 스킵")

    # 모델 로드 (집에서 실행 권장)
    model = get_model()

    buffer_ids: List[str] = []
    buffer_texts: List[str] = []
    buffer_metas: List[Dict[str, Any]] = []

    total_seen = 0
    total_written = 0

    def flush() -> None:
        nonlocal total_written, buffer_ids, buffer_texts, buffer_metas
        if not buffer_ids:
            return

        vectors = model.encode(
            buffer_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        with out_path.open("a", encoding="utf-8") as f:
            for rid, txt, vec, md in zip(buffer_ids, buffer_texts, vectors, buffer_metas):
                rec = {
                    "id": rid,
                    "text": txt,
                    "vector": vec.tolist(),
                    "metadata": md,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_written += 1

        buffer_ids, buffer_texts, buffer_metas = [], [], []

    # 스트리밍으로 쌓다가 배치로 임베딩하고 바로 저장
    for u in iter_units(units_dir):
        total_seen += 1
        rid = make_record_id(u)
        if rid in done_ids:
            continue

        txt = build_embedding_text(u)
        meta = u.get("meta", {}) or {}
        md = {
            "doc_title": meta.get("doc_title"),
            "doc_class": meta.get("doc_class"),
            "source_file": meta.get("source_file"),
            "path": u.get("path"),
            "unit_id": u.get("unit_id"),
        }

        buffer_ids.append(rid)
        buffer_texts.append(txt)
        buffer_metas.append(md)

        # 배치가 차면 바로 처리/저장
        if len(buffer_ids) >= BATCH_SIZE:
            flush()

    # 남은 것 처리
    flush()

    print(f"처리 대상(전체 유닛): {total_seen}개")
    print(f"저장 완료(이번 실행): {total_written}개")
    print(f"출력 파일: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
