import json
import re
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi


def tokenize_kr_mvp(text: str):
    # MVP: 공백 기반 + 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ")


def build_bm25(docs_path: Path, out_path: Path):
    if not docs_path.exists():
        raise FileNotFoundError(f"docs.jsonl not found: {docs_path}")

    texts = []
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            texts.append(o["text"])

    token_corpus = [tokenize_kr_mvp(t) for t in texts]
    bm25 = BM25Okapi(token_corpus)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(bm25, f)

    print("[OK] BM25 build complete")
    print(f" - input : {docs_path}")
    print(f" - output: {out_path}")
    print(f" - docs  : {len(texts)}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]   # .../law_ch
    EMBED_DIR = BASE_DIR / "embedded"

    DOCS_PATH = EMBED_DIR / "docs.jsonl"
    OUT_BM25 = EMBED_DIR / "bm25.pkl"

    build_bm25(DOCS_PATH, OUT_BM25)
