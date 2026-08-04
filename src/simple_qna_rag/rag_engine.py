#!/usr/bin/env python3
"""
RAG 코어 엔진

벡터스토어, LLM, 검색 체인을 관리하는 핵심 모듈
한 번 초기화하면 전역으로 재사용 가능
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from simple_qna_rag.config import (
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
)
from simple_qna_rag.intent_classifier import classify_intent
from simple_qna_rag.prompt_templates import get_template_by_intent


@dataclass
class RetrievalStageTrace:
    """검색 파이프라인 한 단계의 계측 결과.

    name은 "bm25" | "dense" | "rrf" | "mmr" | "reranker" | "total" 중 하나만
    사용한다. latency_ms는 time.perf_counter() 기준 경과 시간(ms)이고,
    candidate_count는 해당 단계가 반환한 문서 수다.
    """

    name: str
    latency_ms: float
    candidate_count: int


@dataclass
class RetrievalTrace:
    """`_retrieve_documents(question, trace=RetrievalTrace())`로 opt-in할 때만
    채워지는 계측 컨테이너. trace=None(기본값)으로 호출하면 이 객체도, 내부의
    RetrievalStageTrace도 전혀 생성되지 않는다(M2-REQ-006, 비활성 시 zero-cost)."""

    stages: list[RetrievalStageTrace] = field(default_factory=list)


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
            # Intent Classifier는 첫 쿼리 시 lazy loading
            self.intent_classifier_loaded = False
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

            # 4. Dense Retriever 설정
            self.dense_retriever = self._setup_retriever()

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
                f"먼저 simple-qna-rag-index를 실행하여 문서를 등록해주세요."
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

    def _setup_retriever(self):
        """Dense Retriever 설정"""
        if USE_HYBRID_SEARCH:
            print(f"🔍 3-Stage Retrieval 파이프라인 설정")
            print(f"   Stage 1 - Hybrid Search:")
            print(f"      - Dense (FAISS): {DENSE_TOP_K}개")
            print(f"      - Sparse (BM25): {BM25_TOP_K}개")
            print(f"      - RRF 융합 후: {RRF_TOP_K}개")

            dense_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": DENSE_TOP_K}
            )

            if USE_MMR:
                print(f"   Stage 2 - MMR (다양성 확보):")
                print(f"      - lambda={MMR_LAMBDA} (0=최대 다양성, 1=최대 관련성)")
                print(f"      - 출력: {MMR_K}개")

            if USE_RERANKER:
                stage_num = 3 if USE_MMR else 2
                print(f"   Stage {stage_num} - Re-ranker:")
                print(f"      - 모델: {RERANKER_MODEL}")
                print(f"      - 최종 출력: {RERANKER_TOP_K}개")

            # 전체 파이프라인 요약
            pipeline_parts = [f"Hybrid({BM25_TOP_K}+{DENSE_TOP_K})", f"RRF({RRF_TOP_K})"]
            if USE_MMR:
                pipeline_parts.append(f"MMR({MMR_K})")
            if USE_RERANKER:
                pipeline_parts.append(f"Re-rank({RERANKER_TOP_K})")
            print(f"   파이프라인: {' → '.join(pipeline_parts)}")

        elif USE_MMR:
            print(f"🔍 Retriever 설정: MMR (다양성 확보)")
            print(f"   - k={MMR_K}, fetch_k={MMR_FETCH_K}, lambda={MMR_LAMBDA}")
            dense_retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": MMR_K,
                    "fetch_k": MMR_FETCH_K,
                    "lambda_mult": MMR_LAMBDA
                }
            )

            if USE_RERANKER:
                print(f"🔍 Re-ranker 활성화:")
                print(f"   - 모델: {RERANKER_MODEL}")
                print(f"   - 최종 문서 수: {RERANKER_TOP_K}")
        else:
            print(f"🔍 Retriever 설정: Similarity (유사도)")
            print(f"   - k={RETRIEVAL_K}")
            dense_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVAL_K}
            )

            if USE_RERANKER:
                print(f"🔍 Re-ranker 활성화:")
                print(f"   - 모델: {RERANKER_MODEL}")
                print(f"   - 최종 문서 수: {RERANKER_TOP_K}")

        print(f"✅ Retriever 설정 완료")
        return dense_retriever

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

    def _apply_mmr(self, query: str, documents, top_k: int = 20, lambda_mult: float = 0.5):
        """
        MMR (Maximal Marginal Relevance) 적용

        이미 검색된 문서들에 대해 다양성을 확보하면서 관련성 높은 문서를 선택합니다.

        Args:
            query: 검색 쿼리
            documents: RRF 등으로 선별된 문서 리스트
            top_k: 최종 선택할 문서 수
            lambda_mult: 관련성 vs 다양성 밸런스 (0=최대 다양성, 1=최대 관련성)

        Returns:
            다양성이 확보된 문서 리스트
        """
        import numpy as np

        if len(documents) <= top_k:
            return documents

        # 임베딩 모델 가져오기
        embeddings_model = self.vectorstore.embeddings

        # 쿼리 임베딩
        query_embedding = np.array(embeddings_model.embed_query(query))

        # 문서 임베딩
        doc_embeddings = np.array([
            embeddings_model.embed_query(doc.page_content) for doc in documents
        ])

        # 코사인 유사도 계산 (정규화된 벡터이므로 내적 = 코사인 유사도)
        query_similarities = np.dot(doc_embeddings, query_embedding)

        selected_indices = []
        remaining_indices = list(range(len(documents)))

        # 첫 번째 문서: 쿼리와 가장 유사한 문서 선택
        first_idx = np.argmax(query_similarities)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # 나머지 문서 선택 (MMR 알고리즘)
        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                # 쿼리와의 유사도
                relevance = query_similarities[idx]

                # 이미 선택된 문서들과의 최대 유사도 (중복성)
                if selected_indices:
                    similarities_to_selected = [
                        np.dot(doc_embeddings[idx], doc_embeddings[sel_idx])
                        for sel_idx in selected_indices
                    ]
                    max_similarity = max(similarities_to_selected)
                else:
                    max_similarity = 0

                # MMR 점수: lambda * 관련성 - (1 - lambda) * 중복성
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
                mmr_scores.append((idx, mmr_score))

            # 가장 높은 MMR 점수를 가진 문서 선택
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [documents[i] for i in selected_indices]

    def _retrieve_documents(
        self,
        question: str,
        trace: "RetrievalTrace | None" = None,
    ):
        """
        문서 검색 (3-Stage Retrieval 파이프라인)

        Hybrid Search 사용 시:
            Stage 1: Hybrid Search (BM25 + FAISS) → RRF 융합
            Stage 2: MMR (다양성 확보) - USE_MMR=True일 때
            Stage 3: Re-ranking (Cross-Encoder) - USE_RERANKER=True일 때

        trace가 None(기본값)이면 계측이 전혀 수행되지 않는다 — 기존 호출자
        (RAGEngine.query() 등)는 시그니처와 반환값이 이전과 100% 동일하다
        (M2-REQ-006/012). trace=RetrievalTrace()를 넘기면 evaluator가 각
        단계의 latency(ms)와 candidate_count를 얻을 수 있다. 검색 로직 자체는
        이 메서드 하나에만 존재하며, trace 유무에 따라 기록 여부만 달라진다.
        """
        t_total0 = time.perf_counter() if trace is not None else None

        def stage(name, fn):
            if trace is None:
                return fn()
            t0 = time.perf_counter()
            result = fn()
            trace.stages.append(
                RetrievalStageTrace(name, (time.perf_counter() - t0) * 1000, len(result))
            )
            return result

        if USE_HYBRID_SEARCH:
            # Stage 1: Hybrid Search + RRF
            bm25_docs = stage(
                "bm25",
                lambda: self.bm25_retriever.invoke(question, top_k=BM25_TOP_K),
            )
            dense_docs = stage(
                "dense",
                lambda: self.dense_retriever.invoke(question) if self.dense_retriever else [],
            )
            docs = stage(
                "rrf",
                lambda: self._reciprocal_rank_fusion(
                    bm25_docs, dense_docs,
                    top_k=RRF_TOP_K,
                    k=RRF_CONSTANT
                ),
            )

            # Stage 2: MMR (다양성 확보)
            if USE_MMR:
                docs = stage(
                    "mmr",
                    lambda: self._apply_mmr(
                        question, docs,
                        top_k=MMR_K,
                        lambda_mult=MMR_LAMBDA
                    ),
                )

            # Stage 3: Re-ranking
            if USE_RERANKER:
                docs = stage(
                    "reranker",
                    lambda: self._rerank_documents(question, docs, top_k=RERANKER_TOP_K),
                )

        elif USE_MMR:
            # MMR만 사용 (Hybrid Search 비활성화 시)
            docs = stage("dense", lambda: self.dense_retriever.invoke(question))
            if USE_RERANKER:
                docs = stage(
                    "reranker",
                    lambda: self._rerank_documents(question, docs, top_k=RERANKER_TOP_K),
                )

        elif USE_RERANKER:
            # Re-ranking만 사용
            docs = stage("dense", lambda: self.dense_retriever.invoke(question))
            docs = stage(
                "reranker",
                lambda: self._rerank_documents(question, docs, top_k=RERANKER_TOP_K),
            )

        else:
            # 기본 Similarity 검색
            docs = stage("dense", lambda: self.dense_retriever.invoke(question))

        if trace is not None:
            trace.stages.append(
                RetrievalStageTrace("total", (time.perf_counter() - t_total0) * 1000, len(docs))
            )

        return docs

    def query(self, question: str) -> Dict[str, any]:
        """
        질문에 대한 답변 생성 (Intent 분류 → 템플릿 선택 → 답변 생성)

        Args:
            question: 사용자 질문

        Returns:
            dict: {
                "answer": str,  # 답변
                "sources": List[Dict],  # 출처 문서
                "success": bool,
                "intent": str  # 분류된 의도
            }
        """
        try:
            if not self.llm:
                raise RuntimeError("RAG 엔진이 초기화되지 않았습니다.")

            # Intent Classifier 첫 로딩 메시지 (한 번만)
            if not self.intent_classifier_loaded:
                print("\n🔧 Intent Classifier 로딩 중...")
                self.intent_classifier_loaded = True

            # 1. Intent 분류
            try:
                intent = classify_intent(question)
                print(f"🎯 Intent 분류 결과: {intent}")
            except FileNotFoundError:
                # Intent Classifier 모델이 없으면 기본 템플릿 사용
                print("⚠️  Intent Classifier 모델을 찾을 수 없습니다. 기본 템플릿을 사용합니다.")
                print("   train_intent_classifier.py를 실행하여 모델을 학습하세요.")
                intent = "other"
            except Exception as e:
                print(f"⚠️  Intent 분류 실패: {e}. 기본 템플릿을 사용합니다.")
                intent = "other"

            # 2. Intent에 맞는 템플릿 선택
            template_str = get_template_by_intent(intent)
            prompt = PromptTemplate(
                template=template_str,
                input_variables=["context", "question"]
            )

            # 3. 문서 검색
            retrieved_docs = self._retrieve_documents(question)

            # 4. 컨텍스트 생성
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)

            # 5. QA 체인 동적 생성 및 실행
            qa_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # 답변 생성
            answer = qa_chain.invoke({"context": context, "question": question})

            # 6. 출처 정보 포맷 (전체 반환, UI에서 제어)
            sources = []
            for i, doc in enumerate(retrieved_docs, 1):
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
                "success": True,
                "intent": intent
            }

        except Exception as e:
            print(f"❌ 질의 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"오류가 발생했습니다: {str(e)}",
                "sources": [],
                "success": False,
                "intent": "error"
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
