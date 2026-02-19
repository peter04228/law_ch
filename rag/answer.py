# rag/answer.py
from typing import Dict, Any
from rag.prompt import SYSTEM, build_user_prompt
from search.retrieve import retrieve

def answer_query(query: str, k: int = 8) -> Dict[str, Any]:
    pack = retrieve(query, k=k)

    # TODO: 여기서 실제 LLM 호출 (OpenAI API든 로컬 LLM이든)
    # messages = [{"role":"system","content":SYSTEM},{"role":"user","content":build_user_prompt(query, pack["context"])}]
    # answer = call_llm(messages)

    answer = "(여기에 LLM 호출 결과가 들어감)"

    return {
        "query": query,
        "answer": answer,
        "citations": pack["citations"],
        "used_context": pack["context"],  # 디버그용(필요 없으면 제거)
    }
