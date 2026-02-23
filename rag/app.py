# app.py
from rag.answer import answer_query
from rag.logger import append_jsonl

LOG_PATH = "logs/qa_log.jsonl"

if __name__ == "__main__":
    q = input("질문: ").strip()
    out = answer_query(q, k=8)

    print("\n[답변]\n", out["answer"])
    print("\n[근거]")
    for c in out["citations"]:
        print("-", c["rank"], c["doc_title"], c["unit_id"], c["path"])

    # 🔽 여기서 바로 로그 저장
    retrieved_slim = []
    for c in out["citations"]:
        retrieved_slim.append({
            "rank": c.get("rank"),
            "doc_title": c.get("doc_title"),
            "unit_id": c.get("unit_id"),
            "path": c.get("path"),
        })

    append_jsonl(LOG_PATH, {
        "question": q,
        "answer": out["answer"],
        "retrieved_docs": retrieved_slim,
    })