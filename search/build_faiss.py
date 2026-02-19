import json
from pathlib import Path
import numpy as np


def build_faiss(
    input_jsonl: Path,
    output_index: Path,
    use_cosine: bool = True,
):
    """
    all_laws.jsonl(벡터 포함) -> faiss.index 생성

    - use_cosine=True:
        문서 벡터를 L2 정규화 후 IndexFlatIP(Inner Product) 사용
        => 코사인 유사도 검색
    """
    import faiss

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input not found: {input_jsonl}")

    # dim 자동 감지
    with input_jsonl.open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    dim = len(first["vector"])

    vectors = []
    count = 0

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            o = json.loads(line)
            v = np.array(o["vector"], dtype=np.float32)

            if v.ndim != 1 or v.shape[0] != dim:
                raise ValueError(
                    f"Vector dim mismatch at line {line_no}: got {v.shape}, expected ({dim},)"
                )

            vectors.append(v)
            count += 1

    X = np.vstack(vectors).astype(np.float32)

    if use_cosine:
        faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(X)

    output_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_index))

    print("[OK] FAISS index build complete")
    print(f" - input : {input_jsonl}")
    print(f" - output: {output_index}")
    print(f" - dim   : {dim}")
    print(f" - ntotal: {index.ntotal}")
    print(f" - metric: {'cosine(IP+norm)' if use_cosine else 'L2'}")


if __name__ == "__main__":
    # ✅ law_ch/search/build_faiss.py 위치 기준으로 law_ch 루트를 계산
    BASE_DIR = Path(__file__).resolve().parents[1]  # .../law_ch
    EMBED_DIR = BASE_DIR / "embedded"

    INPUT_JSONL = EMBED_DIR / "all_laws.jsonl"
    OUTPUT_INDEX = EMBED_DIR / "faiss.index"

    build_faiss(
        input_jsonl=INPUT_JSONL,
        output_index=OUTPUT_INDEX,
        use_cosine=True,
    )
