"""
law_ch.search 패키지

- docs.py            : docs.jsonl 생성
- build_faiss.py     : FAISS 인덱스 생성(벡터 검색 인덱스 생성 => embedded/all_laws.jsonl 저장)
- build_bm25.py      : BM25 인덱스 생성(키워드 검색 인덱스 생성 => embedded/docs.jsonl 저장)
- docs.py            : BM25는 벡터 사용 X(Vector 제거 후 bm25.pki 생성)
- hybrid_search.py   : 하이브리드 검색(FIASS + BM25)
- retrieve.py        : 검색 결과를 llm에 넣기 좋은 근거로 바꾸는 것. 

"""
