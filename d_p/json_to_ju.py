"""
make_units_batch.py (슬림 저장 버전)
- law_ch/json 폴더의 structured JSON들을 읽어서 embedding_units를 생성
- law_ch/json_units 폴더에는 meta + embedding_units만 저장 (structured 제거)

사용법:
  python make_units_batch.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def safe_str(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else ""


def join_nonempty(lines: List[str]) -> str:
    return "\n".join([ln.strip() for ln in lines if ln and ln.strip()]).strip()


def flatten_node_text(node: Dict[str, Any]) -> str:
    """노드 본문 + 자식(호/목 포함) 본문을 재귀적으로 합쳐 텍스트 1덩어리로 만든다."""
    parts: List[str] = []

    t = safe_str(node.get("text"))
    if t:
        parts.append(t)

    for ch in (node.get("children") or []):
        ch_text = flatten_node_text(ch)
        if not ch_text:
            continue

        ch_key = safe_str(ch.get("key"))
        ch_type = safe_str(ch.get("type"))

        # 호/목이면 번호/문자 prefix를 붙여 가독성 보강(선택)
        prefix = ""
        if ch_key and ch_type in ("num_item", "kor_item"):
            prefix = f"{ch_key}. "

        parts.append((prefix + ch_text).strip())

    return join_nonempty(parts)


def make_units_from_structured(structured: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """structured(조/항 트리) → embedding_units(항 단위 평탄화) 생성"""
    units: List[Dict[str, Any]] = []

    for art in structured:
        if not isinstance(art, dict):
            continue
        if safe_str(art.get("type")) != "article":
            continue

        art_key = safe_str(art.get("key"))
        if not art_key:
            continue

        paragraphs = art.get("children") or []

        # 항이 없는 조 → 조 단위 1개
        if not paragraphs:
            text = flatten_node_text(art)
            if text:
                units.append({
                    "unit_id": art_key,
                    "path": [art_key],
                    "text": text,
                    "meta": meta,
                })
            continue

        # 항 단위
        for para in paragraphs:
            if not isinstance(para, dict):
                continue
            if safe_str(para.get("type")) != "paragraph":
                continue

            para_key = safe_str(para.get("key"))
            if not para_key:
                continue

            text = flatten_node_text(para)
            if not text:
                continue

            units.append({
                "unit_id": f"{art_key}_{para_key}",
                "path": [art_key, para_key],
                "text": text,
                "meta": meta,
            })

    return units


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    json_dir = base_dir / "json"
    units_dir = base_dir / "json_units"

    units_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"json 폴더에 파일이 없습니다: {json_dir}")
        return 2

    ok = 0
    for jf in json_files:
        doc = json.loads(jf.read_text(encoding="utf-8"))
        meta = doc.get("meta", {}) or {}
        structured = doc.get("structured", [])

        if not isinstance(structured, list) or not structured:
            print(f"[스킵] structured 없음/비어있음: {jf.name}")
            continue

        units = make_units_from_structured(structured, meta)

        # ✅ json_units에는 structured를 빼고 저장
        slim = {
            "meta": meta,
            "embedding_units": units,
        }

        out_path = units_dir / jf.name
        out_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[완료] {jf.name} → json_units\\{jf.name} (units: {len(units)})")
        ok += 1

    print(f"\n처리 완료: {ok}개 파일")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
