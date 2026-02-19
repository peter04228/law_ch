1️⃣ 프로젝트 소개 (What)

무엇을 하는 프로젝트인가?

어떤 문제를 해결하는가?

왜 필요한가?

2️⃣ 시스템 아키텍처 (How)

전체 파일 구조

데이터 흐름

RAG 구조

하이브리드 검색 구조

3️⃣ 폴더 구조 설명

d_p

search

rag

embedded

4️⃣ 실행 방법 (How to run)

Python 버전

가상환경 생성

requirements 설치

실행 순서

5️⃣ 기술 스택

Python

FAISS

BM25

Sentence-Transformers

OpenAI API

6️⃣ 향후 개선 방향 (Optional)

Cross-encoder re-ranking

법 체계 우선순위 반영

평가 지표 추가

🔥 이제 Flow Chart 만들어준다

README에 바로 넣을 수 있게 Markdown 기반으로 작성해줄게.

📌 전체 시스템 Flow Chart
📄 PDF 법령 파일
        │
        ▼
[d_p] 데이터 전처리
  - PDF → JSON
  - JSON → 조/항 단위 분리
  - Unit → Embedding 생성
        │
        ▼
📂 embedded/
  - all_laws.jsonl
  - docs.jsonl
        │
        ▼
[search] 인덱스 생성
  ├─ BM25 인덱스 생성
  ├─ FAISS 벡터 인덱스 생성
        │
        ▼
사용자 질문 입력
        │
        ▼
Hybrid Search
  ├─ BM25 검색
  ├─ FAISS 검색
  ├─ 스코어 결합
        │
        ▼
retrieve.py
  ├─ 상위 근거 선택
  ├─ context 구성
        │
        ▼
[rag] LLM 호출
  ├─ 프롬프트 생성
  ├─ OpenAI API 호출
        │
        ▼
📌 최종 답변 + 근거 인용

📂 프로젝트 구조 설명 (README용)
law_ch/
├── d_p/                # 데이터 전처리
│   ├── pdf → json 변환
│   ├── json → unit 분리
│   ├── embedding 생성
│
├── search/             # 검색 엔진
│   ├── BM25 인덱스 생성
│   ├── FAISS 인덱스 생성
│   ├── hybrid_search
│   ├── retrieve
│
├── rag/                # 답변 생성
│   ├── answer.py
│   ├── llm.py
│
├── embedded/           # 생성된 인덱스/데이터
│
├── requirements.txt
└── README.md
