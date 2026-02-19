import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]   # law_ch 폴더
EMBED_DIR = BASE_DIR / "embedded"

INPUT_PATH = EMBED_DIR / "all_laws.jsonl"
OUTPUT_PATH = EMBED_DIR / "docs.jsonl"


def build_docs():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with INPUT_PATH.open("r", encoding="utf-8") as fin, \
         OUTPUT_PATH.open("w", encoding="utf-8") as fout:

        for line in fin:
            obj = json.loads(line)

            doc = {
                "doc_id": obj["id"],
                "text": obj["text"],
                "metadata": obj.get("metadata", {})
            }

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

    print(f"[OK] docs.jsonl 생성 완료")
    print(f" - 총 문서 수: {count}")
    print(f" - 저장 위치: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_docs()
