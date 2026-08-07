"""
RAG 시스템 설정 파일

임베딩 모델 선정 기준:
1. 의미적 일관성 (semantic alignment)
2. 긴 문맥 처리 능력 (512+ tokens)
3. 멀티언어/한국어 지원
4. 도메인 일반화
5. 효율성 (속도·메모리)
"""

import os
from pathlib import Path
import warnings


# M2.5 Phase 2: 제품 package가 설치된 위치를 기준으로 checkout root를 계산한다.
# runtime 환경변수/CLI override와 `runtime/` 전환은 Phase 4에서 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_runtime_path(
    env_name: str,
    default_path: Path,
    legacy_path: Path,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    """Resolve `environment > new default > legacy fallback` without mutating files."""
    values = os.environ if environ is None else environ
    configured = values.get(env_name)
    if configured:
        return Path(configured).expanduser().resolve()

    default_path = default_path.resolve()
    legacy_path = legacy_path.resolve()
    if default_path.exists() and legacy_path.exists():
        raise RuntimeError(
            f"새 runtime 경로와 기존 경로가 모두 존재합니다: "
            f"{default_path}, {legacy_path}. 자동 병합하지 않으므로 하나를 명시적으로 선택하세요."
        )
    if default_path.exists() or not legacy_path.exists():
        return default_path

    warnings.warn(
        f"기존 runtime 경로 {legacy_path}를 사용합니다. {default_path}로 이전하세요.",
        FutureWarning,
        stacklevel=2,
    )
    return legacy_path

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
VECTORSTORE_PATH = str(
    resolve_runtime_path(
        "SIMPLE_QNA_RAG_VECTORSTORE_DIR",
        PROJECT_ROOT / "runtime" / "vectorstore",
        PROJECT_ROOT / "vectorstore",
    )
)
COLLECTION_NAME = "document_collection"

# 문서 처리 설정
DATA_DIR = str(
    resolve_runtime_path(
        "SIMPLE_QNA_RAG_DOCUMENTS_DIR",
        PROJECT_ROOT / "runtime" / "documents",
        PROJECT_ROOT / "data",
    )
)
INTENT_MODEL_PATH = str(
    Path(
        os.environ.get(
            "SIMPLE_QNA_RAG_MODEL_DIR",
            PROJECT_ROOT / "models" / "intent_classifier",
        )
    ).expanduser().resolve()
)
TEMPLATES_DIR = str(PROJECT_ROOT / "web" / "templates")
STATIC_DIR = str(PROJECT_ROOT / "web" / "static")
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
RRF_TOP_K = 50  # RRF 융합 후 선택할 문서 개수 (MMR 입력)
RRF_CONSTANT = 60  # RRF 상수 k (일반적으로 60 사용)

# Re-ranker 설정 (3-Stage Retrieval)
# - 1단계: Hybrid Search (BM25 + FAISS) → RRF로 50개
# - 2단계: MMR로 다양성 확보 → 20개
# - 3단계: Cross-Encoder로 정밀 재정렬 후 상위 10개 선택
USE_RERANKER = True  # Re-ranker 사용 여부
# USE_RERANKER = False
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # Cross-Encoder 모델
# 대안 모델:
# "BAAI/bge-reranker-v2-m3" - 멀티언어, 8192 토큰, 높은 정확도
# "cross-encoder/ms-marco-MiniLM-L-6-v2" - 영어 특화, 빠른 속도
# RERANKER_TOP_K = 5  # Re-ranking 후 최종 반환할 문서 개수
RERANKER_TOP_K = 10

# 웹검색 설정
USE_WEB_SEARCH = True  # 웹검색 기능 활성화 여부
WEB_SEARCH_MAX_RESULTS = 3  # 최대 검색 결과 수
WEB_SEARCH_TIMEOUT = 10  # 검색 타임아웃 (초)
WEB_SEARCH_REGION = "kr-kr"  # 검색 지역 (kr-kr: 한국)

# 프롬프트 템플릿 설정
# PROMPT_TEMPLATE = """당신은 주어진 문서 내용을 바탕으로 정확하게 답변하는 AI 어시스턴트입니다.
#
# 지침:
# 1. 주어진 문맥(Context)의 정보만을 사용하여 답변하세요.
# 2. 문맥에 답이 없으면 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 답변하세요.
# 3. 답변은 명확하고 구체적이어야 합니다.
# 4. 가능한 경우 출처를 인용하세요.
# 5. 불확실한 내용은 추측하지 마세요.
#
# 문맥 (Context):
# {context}
#
# 질문 (Question): {question}
#
# 답변 (Answer):"""

PROMPT_TEMPLATE = """당신은 문서 기반 RAG 시스템의 답변을 생성하는 AI 어시스턴트입니다.
 
 [역할 및 규칙]
 
 1. 먼저 질문과 제공된 문맥(Context)을 분석합니다.
 
 2. 주어진 문맥(Context)의 정보만을 사용하여 답변합니다.
  - 답변은 명확하고 구체적이어야 합니다.
  - 답변의 양식은 표로 정리하기 적합한지 여부에 따라 아래 규칙을 따릅니다.
 
 3. 답변 내용을 표로 정리하기 적합한지 판단합니다.
  - 표로 정리하기 적합한지 판단하근 기준은, 여러 개의 항목을 공통된 기준(열/컬럼)으로 비교 및 정리하기에 적합한지 여부입니다.
  - 답변 본문이 한 줄 이하 이거나 혹은 데이터 행이 1개 이하일 때에는 표로 정리하지 않습니다.
  - 이 판단 과정은 답변에 드러내지 마세요. (생각을 말로 설명하지 마세요.)
 
 4. 표로 정리하기에 적합한 답변 내용은 다음 형식의 Markdown 표로 본문을 작성합니다.
  - 첫 줄: 한 문장 핵심 요약. 만약 사용자가 '예', '아니오'로 답변할 수 있는 질문을 했다면 '예' 또는 '아니오' 형태의 답변을 제일 서두에 표현할 것.
  - 그 아래: 표 (헤더 1행 + 데이터 행 2개 이상)
  - 예시 해더: | 항목 | 설명 | 참고 | 와 같은 형태
  - 불필요한 서론 없이 간결하게 작성
  - 표 내용 후 맨 아래에 결론으로 내용을 요약
 
 5. 표로 정리하기에 부적합한 답변 내용은 억지로 표를 만들지 말고, 자연스러운 문단 또는 bullet/번호 리스트 형식으로 답변합니다.
  - 추상적인 개념 설명, 스토리형 질문은 표로 정리하지 말고 자연스러운 평문으로 답합니다.
  - '표로 답변하기 어렵다'와 같은 메타 설명은 하지 않습니다.
 
 6. 항상 제공된 컨텍스트를 우선적으로 사용하고, 근거 없는 내용을 상상하여 추가하지 않습니다.
 
 7. 문맥에 답이 없으면 '제공된 문서에서 관련 정보를 찾을 수 없습니다' 라고 답변합니다.
 
 8. 가능한 경우 출처를 인용합니다.
 
 문맥 (Context):
 {context}
 
 질문 (Question): {question}
 
 답변 (Answer):"""


# ---------------------------------------------------------------------------
# M3 flag helpers (Design.md §3.4)
#
# 환경변수 파싱 규칙: bool은 {"1","true","yes","on"}(casefold)만 True, 그 외는
# False, 미설정은 기본값. 잘못된 열거값은 import 시 ValueError로 즉시 실패한다
# (조용히 기본값으로 되돌아가면 리포트가 실제 실행 설정을 거짓으로 기록하게
# 되기 때문이다).
# ---------------------------------------------------------------------------

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().casefold() in _TRUE_VALUES


def _env_enum(name: str, default: str, allowed: frozenset[str]) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw not in allowed:
        raise ValueError(f"{name}={raw!r}은(는) {sorted(allowed)} 중 하나여야 합니다")
    return raw


# Phase 2 — MMR stored-vector reuse. 기본값 "stored"는 m3-p2a-stored-vector
# 후보가 §4.1 gate를 전부 만족해(회귀 0, MMR 평균 latency 14,349ms -> 8.07ms)
# 채택된 결과다. 회귀 시 SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE=embed로 즉시 롤백
# 가능하다(Design.md §13.3). 검증 실패 시에도 엔진은 자동으로 embed 강등한다.
MMR_VECTOR_SOURCE = _env_enum(
    "SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE", "stored", frozenset({"embed", "stored"})
)
MMR_VECTOR_VALIDATION_SAMPLE = 3
MMR_VECTOR_COSINE_FLOOR = 0.99
MMR_EMBED_CACHE_MAX_ITEMS = 2048

# Phase 3 — routing signal override. 기본값 "True"는 m3-p3a-signal-override
# 후보가 live 76x3 실행에서 §4.1 gate를 전부 만족해(web_search_recall 15/15
# 매 run, document_route_recall 61/61 중앙값, accuracy 75/76 중앙값) 채택된
# 결과다. 회귀 시 SIMPLE_QNA_RAG_ROUTING_SIGNAL_OVERRIDE=0으로 즉시 롤백
# 가능하다(Design.md §13.3).
ROUTING_SIGNAL_OVERRIDE = _env_bool("SIMPLE_QNA_RAG_ROUTING_SIGNAL_OVERRIDE", True)
ROUTING_CORPUS_TOPIC_HINT = _env_bool("SIMPLE_QNA_RAG_ROUTING_CORPUS_TOPIC_HINT", False)
ROUTING_CORPUS_TOPIC_HINT_MAX_ITEMS = 25

# Phase 4 — Intent template mode
ANSWER_TEMPLATE_MODE = _env_enum(
    "SIMPLE_QNA_RAG_ANSWER_TEMPLATE_MODE", "default", frozenset({"intent", "default"})
)
INTENT_CONFIDENCE_FLOOR = 0.0  # 0.0 == 비활성(강등 없음)

# Phase 5 — BM25 tokenizer (조건부)
BM25_TOKENIZER = _env_enum(
    "SIMPLE_QNA_RAG_BM25_TOKENIZER", "whitespace", frozenset({"whitespace", "char2gram", "bge-subword"})
)
