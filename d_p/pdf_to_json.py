"""
pdf_to_json_batch.py
- law_ch/law_pdf 폴더의 PDF들을 일괄 변환하여 law_ch/json 폴더에 JSON으로 저장

사용법:
  python pdf_to_json_batch.py

요구사항:
  python -m pip install pdfplumber
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber


# =========================
# 정규식(법령 구조)
# =========================
RE_ARTICLE_ANYWHERE = re.compile(r"(제\s*\d+\s*조)(?:\s*\(([^)]*)\))?")
RE_ARTICLE_LINE = re.compile(r"^\s*(제\s*\d+\s*조)(?:\s*\(([^)]*)\))?\s*(.*)$")

RE_PARAGRAPH = re.compile(r"^\s*(①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|\(\d+\)|\d+\))\s*(.*)$")
RE_ITEM_NUM = re.compile(r"^\s*(\d+)[\.\)]\s*(.*)$")
RE_ITEM_KOR = re.compile(r"^\s*([가-하])[\.\)]\s*(.*)$")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(l.rstrip() for l in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def norm_key(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


@dataclass
class Node:
    type: str
    key: Optional[str] = None
    title: Optional[str] = None
    text: str = ""
    children: Optional[List["Node"]] = None


def extract_pdf_lines(pdf_path: Path) -> List[str]:
    lines: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            txt = normalize_text(txt)
            if txt:
                lines.extend(txt.splitlines())
            lines.append("")  # 페이지 경계
    return lines


def split_by_articles(lines: List[str]) -> List[List[str]]:
    text = "\n".join(lines)
    matches = list(RE_ARTICLE_ANYWHERE.finditer(text))
    if not matches:
        return [lines]

    chunks: List[List[str]] = []
    starts = [m.start() for m in matches] + [len(text)]
    for i in range(len(starts) - 1):
        part = text[starts[i]:starts[i + 1]].strip()
        if part:
            chunks.append(part.splitlines())
    return chunks


def parse_structured(lines: List[str]) -> List[Node]:
    result: List[Node] = []
    article_chunks = split_by_articles(lines)

    for chunk in article_chunks:
        if not chunk:
            continue

        first = chunk[0].strip()
        m = RE_ARTICLE_LINE.match(first)

        if m:
            art_key = norm_key(m.group(1))
            art_title = m.group(2)
            rest = (m.group(3) or "").strip()
            article = Node("article", art_key, art_title, rest, [])
            body = chunk[1:]
        else:
            joined = " ".join(chunk)
            m2 = RE_ARTICLE_ANYWHERE.search(joined)
            if not m2:
                continue
            article = Node("article", norm_key(m2.group(1)), m2.group(2), "", [])
            body = chunk

        # 조 제목이 다음 줄에 있는 흔한 케이스 대응
        if not article.title and body:
            cand = body[0].strip()
            if cand and len(cand) <= 30 and not RE_PARAGRAPH.match(cand):
                article.title = cand
                body = body[1:]

        cur_para: Optional[Node] = None
        cur_item: Optional[Node] = None

        def add_child(parent: Node, child: Node) -> None:
            if parent.children is None:
                parent.children = []
            parent.children.append(child)

        def append_text(target: Node, txt: str) -> None:
            target.text = f"{target.text}\n{txt}".strip() if target.text else txt

        for raw in body:
            line = raw.strip()
            if not line:
                continue

            mp = RE_PARAGRAPH.match(line)
            if mp:
                para = Node("paragraph", mp.group(1), None, (mp.group(2) or "").strip(), [])
                add_child(article, para)
                cur_para = para
                cur_item = None
                continue

            mn = RE_ITEM_NUM.match(line)
            if mn:
                item = Node("num_item", mn.group(1), None, (mn.group(2) or "").strip(), [])
                add_child(cur_para or article, item)
                cur_item = item
                continue

            mk = RE_ITEM_KOR.match(line)
            if mk:
                sub = Node("kor_item", mk.group(1), None, (mk.group(2) or "").strip(), [])
                add_child(cur_item or cur_para or article, sub)
                cur_item = sub
                continue

            append_text(cur_item or cur_para or article, line)

        result.append(article)

    return result


def build_embedding_units(nodes: List[Node], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    임베딩 단위는 기본적으로 '항' 단위.
    (항이 없으면 조 단위로 생성)
    """
    units: List[Dict[str, Any]] = []

    for art in nodes:
        if not art.children:
            units.append({
                "unit_id": art.key,
                "path": [art.key],
                "text": (art.text or "").strip(),
                "meta": meta,
            })
            continue

        for para in art.children:
            parts = []
            if para.text:
                parts.append(para.text)
            for item in (para.children or []):
                if item.text:
                    parts.append(item.text)

            units.append({
                "unit_id": f"{art.key}_{para.key}",
                "path": [art.key, para.key],
                "text": "\n".join(parts).strip(),
                "meta": meta,
            })

    return units


def infer_doc_class(title: str) -> str:
    if "시행령" in title:
        return "시행령"
    if "시행규칙" in title:
        return "시행규칙"
    if any(k in title for k in ["고시", "훈령", "예규"]):
        return "고시/훈령/예규"
    if any(k in title for k in ["가이드", "안내", "안내서"]):
        return "가이드라인/안내서"
    return "법률"


def convert_one_pdf(pdf_path: Path, out_json_path: Path) -> Dict[str, Any]:
    lines = extract_pdf_lines(pdf_path)
    structured = parse_structured(lines)

    likely_scanned = sum(len(l) for l in lines) < 300
    doc_title = pdf_path.stem
    doc_class = infer_doc_class(doc_title)

    meta = {
        "doc_title": doc_title,
        "doc_class": doc_class,
        "source_file": pdf_path.name,
        "converted_at": datetime.now().isoformat(timespec="seconds"),
        "likely_scanned_pdf": likely_scanned,
    }

    doc = {
        "meta": meta,
        "structured": [asdict(n) for n in structured],
        "embedding_units": build_embedding_units(structured, meta) if not likely_scanned else [],
    }

    if likely_scanned:
        doc["meta"]["warning"] = "텍스트 추출량이 적습니다. 스캔본 PDF이면 OCR이 필요합니다."

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def main() -> int:
    base_dir = Path(__file__).resolve().parent

    # ✅ 여기만 변경: raw_pdf → law_pdf
    pdf_dir = base_dir / "law_pdf"
    json_dir = base_dir / "json"

    pdf_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"law_pdf 폴더에 PDF가 없습니다: {pdf_dir}")
        print("→ PDF를 여기에 넣어주세요.")
        return 2

    ok = 0
    fail = 0
    scanned = 0

    for pdf_path in pdfs:
        out_path = json_dir / f"{pdf_path.stem}.json"
        try:
            doc = convert_one_pdf(pdf_path, out_path)
            ok += 1
            if doc.get("meta", {}).get("likely_scanned_pdf"):
                scanned += 1
                print(f"[경고/스캔의심] {pdf_path.name} -> json\\{out_path.name}")
            else:
                print(f"[완료] {pdf_path.name} -> json\\{out_path.name}")
        except Exception as e:
            fail += 1
            print(f"[실패] {pdf_path.name}: {e}")

    print(f"\n요약: 성공 {ok} / 실패 {fail} / 스캔의심 {scanned}")
    print(f"- 입력: {pdf_dir}")
    print(f"- 출력: {json_dir}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
