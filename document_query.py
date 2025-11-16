#!/usr/bin/env python3
"""
문서 질의 프로그램

벡터스토어에서 관련 문서를 검색하고 LLM을 사용하여 질문에 답변하는 대화형 프로그램
"""

import os
import sys

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from config import (
    EMBEDDING_MODEL_NAME,
    VECTORSTORE_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RETRIEVAL_K,
    USE_MMR,
    MMR_FETCH_K,
    MMR_K,
    MMR_LAMBDA,
    NORMALIZE_EMBEDDINGS,
    USE_HYBRID_SEARCH,
    BM25_TOP_K,
    DENSE_TOP_K,
    RRF_TOP_K,
    RRF_CONSTANT,
    USE_RERANKER,
    RERANKER_MODEL,
    RERANKER_TOP_K,
    PROMPT_TEMPLATE,
)


def create_bm25_retriever(vectorstore: FAISS):
    """
    BM25 기반 Sparse Retriever를 생성합니다.

    Args:
        vectorstore: FAISS 벡터스토어 (문서 로드용)

    Returns:
        BM25Retriever: BM25 검색기
    """
    from rank_bm25 import BM25Okapi

    # FAISS에서 모든 문서 가져오기
    all_docs = vectorstore.docstore._dict.values()
    all_docs = list(all_docs)

    # 문서 텍스트를 토큰화 (간단한 공백 기반)
    tokenized_docs = [doc.page_content.split() for doc in all_docs]

    # BM25 인덱스 생성
    bm25 = BM25Okapi(tokenized_docs)

    print(f"✅ BM25 인덱스 생성 완료 (문서 {len(all_docs)}개)")

    # BM25 검색 함수 래퍼
    class BM25Retriever:
        def __init__(self, bm25_index, documents):
            self.bm25 = bm25_index
            self.documents = documents

        def invoke(self, query: str, top_k: int = 50):
            """질문에 대해 BM25로 상위 문서 반환"""
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)

            # 점수 기준 정렬
            scored_docs = list(zip(self.documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # 상위 top_k개 반환
            top_docs = [doc for doc, score in scored_docs[:top_k]]
            return top_docs

    return BM25Retriever(bm25, all_docs)


def load_vectorstore() -> FAISS:
    """
    저장된 벡터스토어를 로드합니다.

    Returns:
        FAISS: 로드된 벡터스토어
    """
    print(f"📂 벡터스토어 로딩 중: {VECTORSTORE_PATH}")

    if not os.path.exists(VECTORSTORE_PATH):
        print(f"❌ 오류: 벡터스토어가 존재하지 않습니다: {VECTORSTORE_PATH}")
        print(f"ℹ️  먼저 document_register.py를 실행하여 문서를 등록해주세요.")
        sys.exit(1)

    # 임베딩 모델 초기화 (등록 시와 동일한 모델 사용)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': NORMALIZE_EMBEDDINGS}
    )

    # 벡터스토어 로드 (FAISS)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # 로컬 파일 신뢰
    )

    print(f"✅ 벡터스토어 로드 완료 (FAISS IndexFlatIP)")

    return vectorstore


def reciprocal_rank_fusion(bm25_docs, dense_docs, top_k: int = 20, k: int = 60):
    """
    RRF(Reciprocal Rank Fusion)를 사용하여 BM25와 Dense 검색 결과를 융합합니다.

    Args:
        bm25_docs: BM25 검색 결과 문서 리스트
        dense_docs: Dense(FAISS) 검색 결과 문서 리스트
        top_k: 최종 반환할 문서 개수
        k: RRF 상수 (일반적으로 60)

    Returns:
        list: RRF로 융합된 상위 문서들
    """
    from collections import defaultdict

    # 문서별 RRF 점수 계산
    rrf_scores = defaultdict(float)

    # BM25 결과에서 RRF 점수 계산
    for rank, doc in enumerate(bm25_docs, start=1):
        doc_id = id(doc)  # 문서 고유 ID
        rrf_scores[doc_id] += 1.0 / (k + rank)

    # Dense 결과에서 RRF 점수 계산
    for rank, doc in enumerate(dense_docs, start=1):
        doc_id = id(doc)
        rrf_scores[doc_id] += 1.0 / (k + rank)

    # 모든 고유 문서 수집 (중복 제거)
    all_docs = {}
    for doc in bm25_docs + dense_docs:
        all_docs[id(doc)] = doc

    # RRF 점수 기준 정렬
    sorted_docs = sorted(
        all_docs.items(),
        key=lambda x: rrf_scores[x[0]],
        reverse=True
    )

    # 상위 top_k개 반환
    top_docs = [doc for doc_id, doc in sorted_docs[:top_k]]

    print(f"🔀 RRF 융합 완료:")
    print(f"   - BM25: {len(bm25_docs)}개")
    print(f"   - Dense: {len(dense_docs)}개")
    print(f"   - 고유 문서: {len(all_docs)}개")
    print(f"   - 최종 선택: {len(top_docs)}개")

    return top_docs


def rerank_documents(query: str, documents, top_k: int = 5):
    """
    Cross-Encoder를 사용하여 문서를 재정렬합니다.

    Args:
        query: 사용자 질문
        documents: 재정렬할 문서 리스트
        top_k: 반환할 상위 문서 개수

    Returns:
        list: 재정렬된 상위 문서들
    """
    from sentence_transformers import CrossEncoder

    # Cross-Encoder 모델 로드 (캐싱됨)
    if not hasattr(rerank_documents, 'model'):
        print(f"🔧 Re-ranker 모델 로딩 중: {RERANKER_MODEL}")
        rerank_documents.model = CrossEncoder(RERANKER_MODEL, max_length=512)
        print(f"✅ Re-ranker 모델 로드 완료")

    model = rerank_documents.model

    # 문서와 질문 쌍 생성
    pairs = [[query, doc.page_content] for doc in documents]

    # 점수 계산
    scores = model.predict(pairs)

    # 점수 기준 정렬 (내림차순)
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # 상위 top_k개 반환
    top_docs = [doc for doc, score in scored_docs[:top_k]]

    print(f"🔄 Re-ranking 완료: {len(documents)}개 → {len(top_docs)}개 (상위 {top_k}개)")
    for i, (doc, score) in enumerate(scored_docs[:top_k], 1):
        source = doc.metadata.get('source', 'Unknown')
        print(f"   [{i}] Score: {score:.4f} | {source}")

    return top_docs


def setup_qa_chain(vectorstore: FAISS, bm25_retriever=None):
    """
    QA 체인을 설정합니다.
    3-Stage Retrieval: Hybrid Search (BM25 + FAISS) → RRF → Re-ranking

    Args:
        vectorstore: 벡터스토어
        bm25_retriever: BM25 검색기 (하이브리드 검색용, optional)

    Returns:
        tuple: (retriever, qa_chain)
    """
    print(f"🔧 LLM 초기화 중: {OLLAMA_MODEL}")
    print(f"ℹ️  Ollama가 실행 중이고 {OLLAMA_MODEL} 모델이 설치되어 있어야 합니다.")
    print(f"ℹ️  OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")

    # Ollama LLM 초기화
    try:
        llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,  # 낮은 temperature로 일관성 있는 답변 생성
        )
        # 간단한 테스트
        _ = llm.invoke("test")
        print(f"✅ LLM 초기화 완료")
    except Exception as e:
        print(f"❌ LLM 초기화 실패: {e}")
        print(f"\nℹ️  Ollama 설정 확인:")
        print(f"  1. Ollama가 실행 중인지 확인: ollama serve")
        print(f"  2. 모델이 설치되어 있는지 확인: ollama list")
        print(f"  3. 모델 설치: ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

    # 프롬프트 템플릿 설정 (config에서 가져옴)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Dense Retriever 설정 (FAISS)
    if USE_HYBRID_SEARCH:
        print(f"🔍 Stage 1 - Hybrid Search 설정")
        print(f"   - Dense (FAISS): {DENSE_TOP_K}개")
        print(f"   - Sparse (BM25): {BM25_TOP_K}개")
        print(f"   - RRF 융합 후: {RRF_TOP_K}개")

        # FAISS Dense retriever
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": DENSE_TOP_K}
        )
    elif USE_MMR:
        print(f"🔍 Stage 1 - Retriever 설정: MMR (다양성 확보)")
        print(f"   - k={MMR_K}, fetch_k={MMR_FETCH_K}, lambda={MMR_LAMBDA}")
        dense_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": MMR_K,
                "fetch_k": MMR_FETCH_K,
                "lambda_mult": MMR_LAMBDA
            }
        )
    else:
        print(f"🔍 Stage 1 - Retriever 설정: Similarity (유사도)")
        print(f"   - k={RETRIEVAL_K}")
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K}
        )

    # Re-ranker 설정
    stage_num = 2 if USE_HYBRID_SEARCH else 2
    if USE_RERANKER:
        print(f"🔍 Stage {stage_num} - Re-ranker 활성화")
        print(f"   - 모델: {RERANKER_MODEL}")
        print(f"   - 최종 문서 수: {RERANKER_TOP_K}")
        if USE_HYBRID_SEARCH:
            print(f"   - 파이프라인: BM25+FAISS({BM25_TOP_K}+{DENSE_TOP_K}개) → RRF({RRF_TOP_K}개) → Re-rank({RERANKER_TOP_K}개)")
        else:
            print(f"   - 파이프라인: FAISS({MMR_K if USE_MMR else RETRIEVAL_K}개) → Re-rank({RERANKER_TOP_K}개)")

    # QA 체인 생성 (LCEL 방식)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 하이브리드 검색 + Re-ranker 파이프라인
    if USE_HYBRID_SEARCH:
        def hybrid_retrieve_and_rerank(query: str):
            """Hybrid Search (BM25 + FAISS) + Re-ranker 파이프라인"""
            # Stage 1a: BM25 검색
            bm25_docs = bm25_retriever.invoke(query, top_k=BM25_TOP_K)

            # Stage 1b: FAISS Dense 검색
            dense_docs = dense_retriever.invoke(query)

            # Stage 1c: RRF 융합
            fused_docs = reciprocal_rank_fusion(
                bm25_docs, dense_docs,
                top_k=RRF_TOP_K,
                k=RRF_CONSTANT
            )

            # Stage 2: Re-ranking (옵션)
            if USE_RERANKER:
                final_docs = rerank_documents(query, fused_docs, top_k=RERANKER_TOP_K)
            else:
                final_docs = fused_docs

            return final_docs

        qa_chain = (
            {"context": RunnableLambda(hybrid_retrieve_and_rerank) | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        retriever = None  # 하이브리드 검색 시 단일 retriever는 사용 안 함

    # Re-ranker만 사용 (하이브리드 검색 없음)
    elif USE_RERANKER:
        def retrieve_and_rerank(query: str):
            """Dense Retriever + Re-ranker 파이프라인"""
            # Stage 1: FAISS 검색
            docs = dense_retriever.invoke(query)
            # Stage 2: Cross-Encoder Re-ranking
            reranked_docs = rerank_documents(query, docs, top_k=RERANKER_TOP_K)
            return reranked_docs

        qa_chain = (
            {"context": RunnableLambda(retrieve_and_rerank) | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        retriever = dense_retriever

    # 기본 Dense Retriever만 사용
    else:
        qa_chain = (
            {"context": dense_retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        retriever = dense_retriever

    print(f"✅ QA 체인 설정 완료")

    return retriever, qa_chain, bm25_retriever if USE_HYBRID_SEARCH else None


def is_exit_command(user_input: str) -> bool:
    """
    종료 명령어인지 확인합니다.

    Args:
        user_input: 사용자 입력

    Returns:
        bool: 종료 명령어 여부
    """
    exit_keywords = ['종료', '끝', 'stop', 'quit', 'exit', 'finish']
    return user_input.strip().lower() in exit_keywords


def format_source_documents(source_docs):
    """
    출처 문서를 포맷팅합니다.

    Args:
        source_docs: 출처 문서 리스트

    Returns:
        str: 포맷팅된 출처 정보
    """
    if not source_docs:
        return "출처 정보 없음"

    sources = []
    for i, doc in enumerate(source_docs, 1):
        source = doc.metadata.get('source', '알 수 없음')
        page = doc.metadata.get('page', None)

        if page is not None:
            sources.append(f"  [{i}] {source} (페이지 {page + 1})")
        else:
            sources.append(f"  [{i}] {source}")

    return "\n".join(sources)


def run_query_loop(retriever, qa_chain, bm25_retriever=None):
    """
    사용자 질의를 받아 처리하는 메인 루프입니다.

    Args:
        retriever: 문서 검색기 (Dense retriever, 하이브리드 시 None)
        qa_chain: QA 체인
        bm25_retriever: BM25 검색기 (하이브리드 검색 시 사용)
    """
    print("\n" + "=" * 60)
    print("💬 문서 Q&A 시스템에 오신 것을 환영합니다!")
    print("=" * 60)
    print("ℹ️  종료하려면 '종료', '끝', 'stop', 'quit', 'finish' 중 하나를 입력하세요.")
    print("=" * 60 + "\n")

    while True:
        try:
            # 사용자 입력 받기
            user_question = input("🤔 질문을 입력하세요: ").strip()

            # 빈 입력 처리
            if not user_question:
                print("⚠️  질문을 입력해주세요.\n")
                continue

            # 종료 명령어 확인
            if is_exit_command(user_question):
                print("\n👋 문서 Q&A 시스템을 종료합니다. 감사합니다!")
                break

            # 질의 처리
            print("\n🔍 검색 및 답변 생성 중...")

            # 답변 생성
            answer = qa_chain.invoke(user_question)

            # 관련 문서 검색 (출처 표시용)
            if USE_HYBRID_SEARCH:
                # 하이브리드 검색: BM25 + FAISS + RRF
                bm25_docs = bm25_retriever.invoke(user_question, top_k=BM25_TOP_K)
                dense_docs = retriever.invoke(user_question) if retriever else []
                source_docs = reciprocal_rank_fusion(
                    bm25_docs, dense_docs,
                    top_k=RRF_TOP_K,
                    k=RRF_CONSTANT
                )
                if USE_RERANKER:
                    source_docs = rerank_documents(user_question, source_docs, top_k=RERANKER_TOP_K)
            elif USE_RERANKER:
                # Re-ranker만 사용
                source_docs = retriever.invoke(user_question)
                source_docs = rerank_documents(user_question, source_docs, top_k=RERANKER_TOP_K)
            else:
                # 기본 Dense retriever
                source_docs = retriever.invoke(user_question)

            # 답변 출력
            print("\n" + "=" * 60)
            print("📝 답변:")
            print("=" * 60)
            print(answer)

            # 출처 문서 출력
            if source_docs:
                print("\n" + "-" * 60)
                print("📚 참고 문서:")
                print("-" * 60)
                print(format_source_documents(source_docs))

            print("\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            print("다시 시도해주세요.\n")


def main():
    """
    메인 함수: 벡터스토어 로드 -> QA 체인 설정 -> 질의 루프 실행
    """
    print("=" * 60)
    print("📚 문서 질의 프로그램 시작")
    print("=" * 60)

    # 1. 벡터스토어 로드
    vectorstore = load_vectorstore()

    # 2. BM25 검색기 생성 (하이브리드 검색 사용 시)
    bm25_retriever = None
    if USE_HYBRID_SEARCH:
        print(f"\n🔧 BM25 검색기 생성 중...")
        bm25_retriever = create_bm25_retriever(vectorstore)

    # 3. QA 체인 설정
    retriever, qa_chain, bm25_ret = setup_qa_chain(vectorstore, bm25_retriever)
    if bm25_ret:
        bm25_retriever = bm25_ret

    # 4. 질의 루프 실행
    run_query_loop(retriever, qa_chain, bm25_retriever)


if __name__ == "__main__":
    main()
