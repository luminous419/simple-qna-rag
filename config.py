"""
RAG 시스템 설정 파일

임베딩 모델 선정 기준:
1. 의미적 일관성 (semantic alignment)
2. 긴 문맥 처리 능력 (512+ tokens)
3. 멀티언어/한국어 지원
4. 도메인 일반화
5. 효율성 (속도·메모리)
"""

# 임베딩 모델 설정
# BAAI/bge-m3:
# - 8192 토큰 지원 (매우 긴 문맥 처리)
# - 멀티언어 지원 (한국어, 영어, 중국어 등 100+ 언어)
# - 높은 MTEB 벤치마크 점수
# - Retrieval, Re-ranking, Embedding 모두 지원
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# 대안 모델들:
# "intfloat/multilingual-e5-large" - 512 토큰, 멀티언어, 검증된 성능
# "jhgan/ko-sroberta-multitask" - 한국어 특화, 512 토큰

# 벡터 스토어 설정
VECTORSTORE_PATH = "./vectorstore"
COLLECTION_NAME = "document_collection"

# 문서 처리 설정
DATA_DIR = "./data"
CHUNK_SIZE = 1000  # 문서 청크 크기
CHUNK_OVERLAP = 200  # 청크 간 오버랩

# LLM 설정 (Ollama)
OLLAMA_BASE_URL = "http://localhost:11434"

# gpt-oss:20b:
# - 오픈소스 GPT 스타일 모델
# - 20B 파라미터로 높은 성능
# - 긴 문맥 처리 능력
# - 한국어 지원
OLLAMA_MODEL = "gpt-oss:20b"

# 이전 설정 (주석 처리):
# OLLAMA_MODEL = "qwen2.5:7b"
# qwen2.5:
# - 128K 컨텍스트 윈도우 (긴 문서 처리)
# - 뛰어난 한국어 지원
# - 높은 추론 능력
# - 효율적인 메모리 사용

# 대안 모델들:
# "qwen2.5:7b" - 128K 컨텍스트, 뛰어난 한국어, 효율적
# "llama3.1:8b" - 128K 컨텍스트, 좋은 한국어 지원
# "gemma2:9b" - 효율적, 빠른 응답
# "mistral:7b" - 균형잡힌 성능

# RAG 검색 설정
RETRIEVAL_K = 4  # 최종 반환할 문서 개수

# MMR (Maximal Marginal Relevance) 설정
# - 유사한 청크만 몰리는 것을 완화하고 다양성 확보
USE_MMR = True  # MMR 검색 사용 여부
MMR_FETCH_K = 100  # 초기 검색할 문서 개수 (후보군)
MMR_K = 20  # MMR 알고리즘으로 재정렬할 문서 개수
MMR_LAMBDA = 0.5  # 다양성 vs 관련성 밸런스 (0=최대 다양성, 1=최대 관련성)

# 벡터 정규화 설정
NORMALIZE_EMBEDDINGS = True  # L2 정규화 사용 (Cosine 유사도 최적화)
# NORMALIZE_EMBEDDINGS = False

# 하이브리드 검색 설정 (Sparse + Dense Retrieval)
# - Sparse: BM25 (키워드 기반)
# - Dense: FAISS (의미 기반)
# - Fusion: RRF (Reciprocal Rank Fusion)
USE_HYBRID_SEARCH = True  # 하이브리드 검색 사용 여부
# USE_HYBRID_SEARCH = False
BM25_TOP_K = 50  # BM25에서 검색할 문서 개수
DENSE_TOP_K = 50  # FAISS에서 검색할 문서 개수
RRF_TOP_K = 20  # RRF 융합 후 최종 선택할 문서 개수
RRF_CONSTANT = 60  # RRF 상수 k (일반적으로 60 사용)

# Re-ranker 설정 (3-Stage Retrieval)
# - 1단계: Hybrid Search (BM25 + FAISS) → RRF로 20개
# - 2단계: MMR로 다양성 확보 (선택적)
# - 3단계: Cross-Encoder로 정밀 재정렬 후 상위 N개 선택
USE_RERANKER = True  # Re-ranker 사용 여부
# USE_RERANKER = False
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # Cross-Encoder 모델
# 대안 모델:
# "BAAI/bge-reranker-v2-m3" - 멀티언어, 8192 토큰, 높은 정확도
# "cross-encoder/ms-marco-MiniLM-L-6-v2" - 영어 특화, 빠른 속도
# RERANKER_TOP_K = 5  # Re-ranking 후 최종 반환할 문서 개수
RERANKER_TOP_K = 10

# 프롬프트 템플릿 설정
PROMPT_TEMPLATE = """당신은 주어진 문서 내용을 바탕으로 정확하게 답변하는 AI 어시스턴트입니다.

지침:
1. 주어진 문맥(Context)의 정보만을 사용하여 답변하세요.
2. 문맥에 답이 없으면 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 답변하세요.
3. 답변은 명확하고 구체적이어야 합니다.
4. 가능한 경우 출처를 인용하세요.
5. 불확실한 내용은 추측하지 마세요.

문맥 (Context):
{context}

질문 (Question): {question}

답변 (Answer):"""
