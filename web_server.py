#!/usr/bin/env python3
"""
FastAPI 기반 웹 서버

RAG 엔진을 백엔드로 사용하는 웹 인터페이스 제공
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from rag_engine import get_rag_engine

# FastAPI 앱 생성
app = FastAPI(
    title="Simple Q&A RAG System",
    description="문서 질의응답 시스템 웹 인터페이스",
    version="1.0.0"
)

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# RAG 엔진 초기화 (앱 시작 시 한 번만)
rag_engine = None


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 RAG 엔진 초기화"""
    global rag_engine
    print("\n🚀 웹 서버 시작 중...")
    try:
        rag_engine = get_rag_engine()
        print("✅ RAG 엔진 로드 완료")
    except Exception as e:
        print(f"❌ RAG 엔진 초기화 실패: {e}")
        raise


# Request/Response 모델
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list
    success: bool


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/rag", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    """
    RAG 질의 API

    Args:
        request: QueryRequest (question: str)

    Returns:
        QueryResponse: {
            answer: str,
            sources: list,
            success: bool
        }
    """
    if not rag_engine:
        return JSONResponse(
            status_code=500,
            content={
                "answer": "RAG 엔진이 초기화되지 않았습니다.",
                "sources": [],
                "success": False
            }
        )

    print(f"\n📝 질문 수신: {request.question}")

    # RAG 엔진 호출
    result = rag_engine.query(request.question)

    print(f"✅ 답변 생성 완료")

    return QueryResponse(**result)


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "rag_engine_initialized": rag_engine is not None
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """웹 서버 시작"""
    print(f"\n🌐 웹 서버 시작: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
