"""Lightweight Web CLI that handles --help before importing the FastAPI app."""

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simple-qna-rag-web",
        description="Simple Q&A RAG FastAPI 서버를 실행합니다.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="바인딩 host")
    parser.add_argument("--port", type=int, default=8000, help="바인딩 port")
    parser.add_argument("--documents-dir", help="원본 문서 경로 override")
    parser.add_argument("--vectorstore-dir", help="vectorstore 경로 override")
    parser.add_argument("--model-dir", help="intent classifier 모델 경로 override")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.documents_dir:
        os.environ["SIMPLE_QNA_RAG_DOCUMENTS_DIR"] = args.documents_dir
    if args.vectorstore_dir:
        os.environ["SIMPLE_QNA_RAG_VECTORSTORE_DIR"] = args.vectorstore_dir
    if args.model_dir:
        os.environ["SIMPLE_QNA_RAG_MODEL_DIR"] = args.model_dir
    from simple_qna_rag.web.server import start_server

    start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
