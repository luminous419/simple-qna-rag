# Q&A 시스템 코드 분석 및 개선 방향 정리

------------------------------------------------------------------------

## 1. 현재 시스템 구조 요약

### (1) 문서 인덱싱

-   `document_register.py`
    -   PDF/txt 로딩
    -   `RecursiveCharacterTextSplitter(1000/200)`
    -   Embedding: **BAAI/bge-m3**
    -   벡터스토어: **FAISS(L2 normalized)**
    -   진행률 표시 스레드

### (2) RAGEngine (`rag_engine.py`)

-   BM25(splitted content)
-   Dense retriever (bge-m3)
-   Hybrid RRF (50+50 → 50)
-   MMR (0.5, 100→20)
-   Cross-Encoder reranker (**BAAI/bge-reranker-v2-m3**)
-   LLM (**Ollama gpt-oss:20b**)
-   Intent → Prompt Template 매핑
-   최종 문서 \~10개를 컨텍스트로 LLM 호출

### (3) Intent Classifier

-   모델: bge-m3 embedding + Linear Softmax
-   라벨: explanation, comparison, procedure, yesno, other, uncertain
-   train.jsonl 약 1200개
-   confidence 기반 fallback 기능 존재

### (4) Prompt Templates

-   intent별로 구조화된 템플릿
-   "문서에 없으면 모른다고 말하기" 안전장치 존재

### (5) Web / CLI

-   FastAPI 기반 웹 UI
-   CLI 테스트 스크립트

------------------------------------------------------------------------

## 2. 강점

-   Hybrid + MMR + Re-ranker까지 포함된 **풀옵션 RAG 파이프라인**
-   Intent 기반 prompting 구조가 명확
-   RAGEngine 싱글톤 설계로 효율적
-   인덱싱/검색/템플릿/웹 구조 분리 매우 깔끔
-   실사용 UX 디테일도 좋음

------------------------------------------------------------------------

## 3. 개선 방향 (코드 수정 X, 아키텍처 방향성 중심)

### A. Retrieval & Ranking 고도화

1.  한국어 BM25 토크나이저 개선 (형태소 분석 기반)
2.  질의 길이·유형에 따른 동적 top_k 조정
3.  문서 단위 vs chunk 단위 점수 결합
4.  Re-ranker 점수 기반 threshold·LLM 신뢰도 전달

### B. 인덱싱 & 문서 구조 개선

1.  메타데이터 스키마 표준화\
    (`source_id`, `section`, `tags`, `page` 등)
2.  증분 인덱싱 / 인덱스 버전 관리(blue-green)
3.  heading 기반 chunking 및 문서 타입별 chunk 전략

### C. Intent Classifier 확장

1.  Intent에 따라 retrieval/LLM 세팅까지 변경
2.  confidence 기반 fallback
3.  로그 기반 재학습(Active learning)

### D. LLM 응답 구조 개선

1.  JSON 출력 강제화\
2.  Streaming + intent별 token policy
3.  Query rewriting 레이어 고려

### E. 운영/아키텍처 개선

1.  logging 모듈 기반 structured log\
    (retrieval latency per stage)
2.  config → BaseSettings로 분리
3.  RAG 평가 세트 구축(hit rate, correctness)
4.  중복 코드(document_query.py) 정리

### F. UX 기능 확장

1.  Web UI에서 문서/출처 하이라이트
2.  Hybrid/MMR/Reranker on/off 실험 옵션
3.  pure-search 모드 제공

------------------------------------------------------------------------

## 4. 단기 개선 우선순위 (쉽게 바로 가능한 것)

1.  로그/설정 개선\
2.  한국어 BM25 토크나이저 개선\
3.  Intent confidence threshold 적용\
4.  RAG 평가 세트 구축\
5.  UI에서 retrieval 문서 시각화 개선

------------------------------------------------------------------------
