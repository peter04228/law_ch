# app.py
from rag.answer import answer_query

if __name__ == "__main__":
    q = input("질문: ").strip()
    out = answer_query(q, k=8)
    print("\n[답변]\n", out["answer"])
    print("\n[근거]")
    for c in out["citations"]:
        print("-", c["rank"], c["doc_title"], c["unit_id"], c["path"])
