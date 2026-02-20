# rag/answer.py
from __future__ import annotations

from typing import Dict, Any

from openai import OpenAI

from rag.prompt import SYSTEM, build_user_prompt
from search.retrieve import retrieve


# 🔴 여기 네 키 넣어라
OPENAI_API_KEY = ""
MODEL_NAME = "gpt-4.1-mini"


def _call_llm(system: str, user: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        truncation="auto",
    )

    return resp.output_text


def answer_query(query: str, k: int = 8) -> Dict[str, Any]:
    pack = retrieve(query, k=k)

    user_prompt = build_user_prompt(query, pack["context"])
    answer = _call_llm(SYSTEM, user_prompt)

    return {
        "query": query,
        "answer": answer,
        "citations": pack["citations"],
        "used_context": pack["context"],
    }