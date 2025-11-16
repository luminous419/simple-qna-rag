#!/usr/bin/env python3
"""
문서 등록 프로그램

data 디렉토리의 PDF 및 텍스트 문서를 벡터스토어에 저장하는 배치 프로그램
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBEDDING_MODEL_NAME,
    VECTORSTORE_PATH,
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    NORMALIZE_EMBEDDINGS,
)


def load_documents() -> List:
    """
    data 디렉토리에서 모든 PDF와 텍스트 파일을 로드합니다.

    Returns:
        List: 로드된 문서 리스트
    """
    print(f"📂 문서 로딩 중: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print(f"⚠️  경고: {DATA_DIR} 디렉토리가 존재하지 않습니다.")
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"✅ {DATA_DIR} 디렉토리를 생성했습니다.")
        print(f"ℹ️  문서를 {DATA_DIR}에 추가하고 다시 실행해주세요.")
        return []

    documents = []

    # PDF 파일 로드
    try:
        pdf_loader = DirectoryLoader(
            DATA_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        print(f"✅ PDF 파일 {len(pdf_docs)}개 로드 완료")
    except Exception as e:
        print(f"⚠️  PDF 로딩 중 오류: {e}")

    # 텍스트 파일 로드
    try:
        text_loader = DirectoryLoader(
            DATA_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
        )
        text_docs = text_loader.load()
        documents.extend(text_docs)
        print(f"✅ 텍스트 파일 {len(text_docs)}개 로드 완료")
    except Exception as e:
        print(f"⚠️  텍스트 파일 로딩 중 오류: {e}")

    print(f"📊 총 {len(documents)}개 문서 로드 완료")
    return documents


def split_documents(documents: List) -> List:
    """
    문서를 청크로 분할합니다.

    Args:
        documents: 분할할 문서 리스트

    Returns:
        List: 분할된 청크 리스트
    """
    print(f"✂️  문서 분할 중 (청크 크기: {CHUNK_SIZE}, 오버랩: {CHUNK_OVERLAP})")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✅ {len(chunks)}개 청크로 분할 완료")

    return chunks


class ProgressTracker:
    """임베딩 진행 상황을 추적하는 클래스"""

    def __init__(self, total_items: int):
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = None
        self.stop_flag = False
        self.lock = threading.Lock()

    def start(self):
        """진행 표시 시작"""
        self.start_time = time.time()
        self.stop_flag = False

        def print_progress():
            """1초마다 dot(.)과 진행률 출력"""
            last_percentage = -1
            while not self.stop_flag:
                with self.lock:
                    if self.processed_items > 0:
                        percentage = int((self.processed_items / self.total_items) * 100)

                        # 퍼센트가 변경되었을 때만 출력
                        if percentage != last_percentage:
                            elapsed = time.time() - self.start_time
                            print(f"\r진행: {'.' * (percentage // 5)} {percentage}% ({self.processed_items}/{self.total_items} 청크) - {elapsed:.1f}초 경과", end='', flush=True)
                            last_percentage = percentage

                time.sleep(1)

        # 백그라운드 스레드에서 진행률 출력
        self.thread = threading.Thread(target=print_progress, daemon=True)
        self.thread.start()

    def update(self, processed: int):
        """처리된 아이템 수 업데이트"""
        with self.lock:
            self.processed_items = processed

    def stop(self):
        """진행 표시 중지"""
        self.stop_flag = True
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        print()  # 새 줄로 이동


def create_vectorstore(chunks: List) -> FAISS:
    """
    벡터스토어를 생성하고 청크를 임베딩하여 저장합니다.
    L2 정규화 + FAISS IndexFlatIP (Inner Product) 사용

    Args:
        chunks: 저장할 청크 리스트

    Returns:
        FAISS: 생성된 벡터스토어
    """
    print(f"🔧 임베딩 모델 초기화 중: {EMBEDDING_MODEL_NAME}")
    print(f"   - L2 정규화: {NORMALIZE_EMBEDDINGS}")

    # HuggingFace 임베딩 모델 초기화
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},  # GPU 사용 시 'cuda'로 변경
        encode_kwargs={'normalize_embeddings': NORMALIZE_EMBEDDINGS}  # L2 정규화
    )

    print(f"💾 벡터스토어 생성 중 (FAISS IndexFlatIP): {VECTORSTORE_PATH}")
    print(f"📊 총 {len(chunks)}개 청크 임베딩 시작...")

    # 기존 벡터스토어가 있다면 삭제
    if os.path.exists(VECTORSTORE_PATH):
        print(f"⚠️  기존 벡터스토어를 삭제합니다: {VECTORSTORE_PATH}")
        import shutil
        shutil.rmtree(VECTORSTORE_PATH)

    # Progress Tracker 시작
    progress = ProgressTracker(len(chunks))
    progress.start()

    try:
        # 배치 단위로 임베딩하여 진행률 추적
        batch_size = 10  # 배치 크기
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk.page_content for chunk in batch]

            # 배치 임베딩
            batch_embeds = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeds)

            # 진행률 업데이트
            progress.update(min(i + batch_size, len(chunks)))

        progress.stop()

        # FAISS 벡터스토어 생성
        print("🔧 FAISS 인덱스 생성 중...")
        import numpy as np
        from langchain_core.documents import Document

        # 임베딩을 numpy 배열로 변환
        embeddings_array = np.array(all_embeddings).astype('float32')

        # FAISS 인덱스 생성
        import faiss
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product 인덱스
        index.add(embeddings_array)

        # FAISS vectorstore 생성
        from langchain_community.docstore.in_memory import InMemoryDocstore

        # 문서 ID 매핑
        index_to_id = {i: str(i) for i in range(len(chunks))}
        docstore = InMemoryDocstore({str(i): chunks[i] for i in range(len(chunks))})

        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_id
        )

    except Exception as e:
        progress.stop()
        raise e

    # 로컬 디렉토리에 저장
    print("💾 벡터스토어 저장 중...")
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    print(f"✅ 벡터스토어 생성 완료: {len(chunks)}개 청크 저장됨")
    print(f"   - 저장 위치: {VECTORSTORE_PATH}")
    print(f"   - 인덱스 타입: FAISS IndexFlatIP (정규화된 벡터 + Inner Product)")

    return vectorstore


def main():
    """
    메인 함수: 문서 로드 -> 분할 -> 벡터스토어 생성 -> 저장
    """
    print("=" * 60)
    print("📚 문서 등록 프로그램 시작")
    print("=" * 60)

    # 1. 문서 로드
    documents = load_documents()

    if not documents:
        print("\n⚠️  처리할 문서가 없습니다.")
        print(f"ℹ️  {DATA_DIR} 디렉토리에 PDF 또는 TXT 파일을 추가해주세요.")
        sys.exit(0)

    # 2. 문서 분할
    chunks = split_documents(documents)

    if not chunks:
        print("\n⚠️  문서 분할 결과가 없습니다.")
        sys.exit(1)

    # 3. 벡터스토어 생성 및 저장
    try:
        vectorstore = create_vectorstore(chunks)

        print("\n" + "=" * 60)
        print("✅ 문서 등록 완료!")
        print("=" * 60)
        print(f"📊 통계:")
        print(f"  - 총 문서 수: {len(documents)}")
        print(f"  - 총 청크 수: {len(chunks)}")
        print(f"  - 저장 위치: {VECTORSTORE_PATH}")
        print(f"  - 임베딩 모델: {EMBEDDING_MODEL_NAME}")
        print("\n💡 이제 document_query.py를 실행하여 질의할 수 있습니다.")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
