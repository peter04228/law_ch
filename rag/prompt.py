# rag/prompt.py
SYSTEM = """너는 식약처 법령/규정 기반 질의응답 도우미다.
반드시 제공된 [근거] 내용 안에서만 답변하라.
근거가 부족하면 '근거 부족'이라고 말하고, 추가로 필요한 정보를 질문하라.
답변 끝에 근거 번호([근거1] 등)를 반드시 명시하라.
"""

def build_user_prompt(query: str, context: str) -> str:
    return f"""아래 근거를 바탕으로 질문에 답해라.

{context}

[질문]
{query}
"""
