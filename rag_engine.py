#!/usr/bin/env python3
"""
RAG 코어 엔진

벡터스토어, LLM, 검색 체인을 관리하는 핵심 모듈
한 번 초기화하면 전역으로 재사용 가능
"""

import os
import sys
from typing import Optional, Dict, List

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


class RAGEngine:
    """RAG 엔진 싱글톤 클래스"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """초기화는 한 번만 수행"""
        if not RAGEngine._initialized:
            self.vectorstore = None
            self.bm25_retriever = None
            self.dense_retriever = None
            self.llm = None
            self.qa_chain = None
            RAGEngine._initialized = True

    def initialize(self) -> bool:
        """
        RAG 엔진 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        try:
            print("=" * 60)
            print("🚀 RAG 엔진 초기화 시작")
            print("=" * 60)

            # 1. 벡터스토어 로드
            self.vectorstore = self._load_vectorstore()

            # 2. BM25 검색기 생성 (하이브리드 검색 사용 시)
            if USE_HYBRID_SEARCH:
                print(f"\n🔧 BM25 검색기 생성 중...")
                self.bm25_retriever = self._create_bm25_retriever(self.vectorstore)

            # 3. LLM 초기화
            self.llm = self._initialize_llm()

            # 4. QA 체인 설정
            self.dense_retriever, self.qa_chain = self._setup_qa_chain()

            print("\n" + "=" * 60)
            print("✅ RAG 엔진 초기화 완료")
            print("=" * 60)
            return True

        except Exception as e:
            print(f"❌ RAG 엔진 초기화 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_vectorstore(self) -> FAISS:
        """벡터스토어 로드"""
        print(f"📂 벡터스토어 로딩 중: {VECTORSTORE_PATH}")

        if not os.path.exists(VECTORSTORE_PATH):
            raise FileNotFoundError(
                f"벡터스토어가 존재하지 않습니다: {VECTORSTORE_PATH}\n"
                f"먼저 document_register.py를 실행하여 문서를 등록해주세요."
            )

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': NORMALIZE_EMBEDDINGS}
        )

        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print(f"✅ 벡터스토어 로드 완료 (FAISS IndexFlatIP)")
        return vectorstore

    def _create_bm25_retriever(self, vectorstore: FAISS):
        """BM25 검색기 생성"""
        from rank_bm25 import BM25Okapi

        all_docs = list(vectorstore.docstore._dict.values())
        tokenized_docs = [doc.page_content.split() for doc in all_docs]
        bm25 = BM25Okapi(tokenized_docs)

        print(f"✅ BM25 인덱스 생성 완료 (문서 {len(all_docs)}개)")

        class BM25Retriever:
            def __init__(self, bm25_index, documents):
                self.bm25 = bm25_index
                self.documents = documents

            def invoke(self, query: str, top_k: int = 50):
                tokenized_query = query.split()
                scores = self.bm25.get_scores(tokenized_query)
                scored_docs = list(zip(self.documents, scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                return [doc for doc, score in scored_docs[:top_k]]

        return BM25Retriever(bm25, all_docs)

    def _initialize_llm(self):
        """LLM 초기화"""
        print(f"🔧 LLM 초기화 중: {OLLAMA_MODEL}")
        print(f"ℹ️  Ollama가 실행 중이고 {OLLAMA_MODEL} 모델이 설치되어 있어야 합니다.")
        print(f"ℹ️  OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")

        llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )

        # 간단한 테스트
        _ = llm.invoke("test")
        print(f"✅ LLM 초기화 완료")

        return llm

    def _setup_qa_chain(self):
        """QA 체인 설정"""
        # 프롬프트 템플릿
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # Dense Retriever 설정
        if USE_HYBRID_SEARCH:
            print(f"🔍 Stage 1 - Hybrid Search 설정")
            print(f"   - Dense (FAISS): {DENSE_TOP_K}개")
            print(f"   - Sparse (BM25): {BM25_TOP_K}개")
            print(f"   - RRF 융합 후: {RRF_TOP_K}개")

            dense_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": DENSE_TOP_K}
            )
        elif USE_MMR:
            print(f"🔍 Stage 1 - Retriever 설정: MMR (다양성 확보)")
            print(f"   - k={MMR_K}, fetch_k={MMR_FETCH_K}, lambda={MMR_LAMBDA}")
            dense_retriever = self.vectorstore.as_retriever(
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
            dense_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVAL_K}
            )

        # Re-ranker 설정
        if USE_RERANKER:
            print(f"🔍 Stage 2 - Re-ranker 활성화")
            print(f"   - 모델: {RERANKER_MODEL}")
            print(f"   - 최종 문서 수: {RERANKER_TOP_K}")
            if USE_HYBRID_SEARCH:
                print(f"   - 파이프라인: BM25+FAISS({BM25_TOP_K}+{DENSE_TOP_K}개) → RRF({RRF_TOP_K}개) → Re-rank({RERANKER_TOP_K}개)")
            else:
                print(f"   - 파이프라인: FAISS({MMR_K if USE_MMR else RETRIEVAL_K}개) → Re-rank({RERANKER_TOP_K}개)")

        # QA 체인 생성
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        if USE_HYBRID_SEARCH:
            def hybrid_retrieve_and_rerank(query: str):
                bm25_docs = self.bm25_retriever.invoke(query, top_k=BM25_TOP_K)
                dense_docs = dense_retriever.invoke(query)
                fused_docs = self._reciprocal_rank_fusion(
                    bm25_docs, dense_docs,
                    top_k=RRF_TOP_K,
                    k=RRF_CONSTANT
                )
                if USE_RERANKER:
                    return self._rerank_documents(query, fused_docs, top_k=RERANKER_TOP_K)
                return fused_docs

            qa_chain = (
                {"context": RunnableLambda(hybrid_retrieve_and_rerank) | RunnableLambda(format_docs), "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

        elif USE_RERANKER:
            def retrieve_and_rerank(query: str):
                docs = dense_retriever.invoke(query)
                return self._rerank_documents(query, docs, top_k=RERANKER_TOP_K)

            qa_chain = (
                {"context": RunnableLambda(retrieve_and_rerank) | RunnableLambda(format_docs), "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

        else:
            qa_chain = (
                {"context": dense_retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

        print(f"✅ QA 체인 설정 완료")
        return dense_retriever, qa_chain

    def _reciprocal_rank_fusion(self, bm25_docs, dense_docs, top_k: int = 20, k: int = 60):
        """RRF 융합"""
        from collections import defaultdict

        rrf_scores = defaultdict(float)

        for rank, doc in enumerate(bm25_docs, start=1):
            rrf_scores[id(doc)] += 1.0 / (k + rank)

        for rank, doc in enumerate(dense_docs, start=1):
            rrf_scores[id(doc)] += 1.0 / (k + rank)

        all_docs = {}
        for doc in bm25_docs + dense_docs:
            all_docs[id(doc)] = doc

        sorted_docs = sorted(
            all_docs.items(),
            key=lambda x: rrf_scores[x[0]],
            reverse=True
        )

        return [doc for doc_id, doc in sorted_docs[:top_k]]

    def _rerank_documents(self, query: str, documents, top_k: int = 5):
        """Cross-Encoder Re-ranking"""
        from sentence_transformers import CrossEncoder

        if not hasattr(self, 'reranker_model'):
            print(f"🔧 Re-ranker 모델 로딩 중: {RERANKER_MODEL}")
            self.reranker_model = CrossEncoder(RERANKER_MODEL, max_length=512)
            print(f"✅ Re-ranker 모델 로드 완료")

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker_model.predict(pairs)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_k]]

    def query(self, question: str) -> Dict[str, any]:
        """
        질문에 대한 답변 생성

        Args:
            question: 사용자 질문

        Returns:
            dict: {
                "answer": str,  # 답변
                "sources": List[Dict],  # 출처 문서
                "success": bool
            }
        """
        try:
            if not self.qa_chain:
                raise RuntimeError("RAG 엔진이 초기화되지 않았습니다.")

            # 답변 생성
            answer = self.qa_chain.invoke(question)

            # 출처 문서 검색
            if USE_HYBRID_SEARCH:
                bm25_docs = self.bm25_retriever.invoke(question, top_k=BM25_TOP_K)
                dense_docs = self.dense_retriever.invoke(question) if self.dense_retriever else []
                source_docs = self._reciprocal_rank_fusion(
                    bm25_docs, dense_docs,
                    top_k=RRF_TOP_K,
                    k=RRF_CONSTANT
                )
                if USE_RERANKER:
                    source_docs = self._rerank_documents(question, source_docs, top_k=RERANKER_TOP_K)
            elif USE_RERANKER:
                source_docs = self.dense_retriever.invoke(question)
                source_docs = self._rerank_documents(question, source_docs, top_k=RERANKER_TOP_K)
            else:
                source_docs = self.dense_retriever.invoke(question)

            # 출처 정보 포맷
            sources = []
            for i, doc in enumerate(source_docs, 1):
                source = doc.metadata.get('source', '알 수 없음')
                page = doc.metadata.get('page', None)
                sources.append({
                    "index": i,
                    "source": source,
                    "page": page + 1 if page is not None else None,
                    "content": doc.page_content[:200]  # 처음 200자만
                })

            return {
                "answer": answer,
                "sources": sources,
                "success": True
            }

        except Exception as e:
            print(f"❌ 질의 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"오류가 발생했습니다: {str(e)}",
                "sources": [],
                "success": False
            }


# 전역 RAG 엔진 인스턴스
_rag_engine = None


def get_rag_engine() -> RAGEngine:
    """RAG 엔진 싱글톤 인스턴스 반환"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
        if not _rag_engine.initialize():
            raise RuntimeError("RAG 엔진 초기화 실패")
    return _rag_engine
