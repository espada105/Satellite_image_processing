from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import sys
import os
from pathlib import Path

# 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.database.search_vector_db_v2 import search_by_text, hybrid_search
from scripts.rag.rag_tool import RAGTool
from scripts.agent.satellite_agent import SatelliteAgent
from scripts.utils.config import PROCESSED_DATA_DIR


app = FastAPI(title="Satellite Geospatial RAG API", version="0.2.0")

# 정적 파일/이미지 마운트
web_dir = Path(__file__).parent / "web"
static_dir = web_dir / "static"
templates_dir = web_dir / "templates"
images_dir = Path(PROCESSED_DATA_DIR)

static_dir.mkdir(parents=True, exist_ok=True)
templates_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 24
    threshold: float = 0.0


class HybridSearchRequest(BaseModel):
    query: Optional[str] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    top_k: int = 24


class RAGRequest(BaseModel):
    query: str
    top_k: int = 24
    metadata_filters: Optional[Dict[str, Any]] = None


class AgentQueryRequest(BaseModel):
    question: str
    model_name: str = "gpt-4o-mini"


# 전역 도구/에이전트
rag_tool = RAGTool(top_k=24)
agent_instance: Optional[SatelliteAgent] = None


@app.get("/", response_class=HTMLResponse)
async def root_page(request: Request) -> HTMLResponse:
    # 매우 간단한 HTML (템플릿 파일이 없더라도 동작)
    # 별도 템플릿 파일은 scripts/api/web/templates/index.html 로 생성 예정
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Satellite Chat</title>"
        "<link rel='stylesheet' href='/static/style.css'>"
        "</head><body>"
        "<div id='app'>"
        "  <h1>Satellite Geospatial Chat</h1>"
        "  <div id='chat'>"
        "    <div id='messages'></div>"
        "    <div id='inputBox'>"
        "      <input id='userInput' placeholder='질문을 입력하세요 (예: flooded area near river)'>"
        "      <button id='sendBtn'>Send</button>"
        "    </div>"
        "  </div>"
        "  <div id='results'><h3>검색 결과</h3><div id='grid'></div></div>"
        "</div>"
        "<script src='/static/app.js'></script>"
        "</body></html>"
    )
    return HTMLResponse(content=html)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/search/text")
async def search_text(req: TextSearchRequest) -> Dict[str, Any]:
    results = search_by_text(req.query, top_k=req.top_k, threshold=req.threshold)
    return {"query": req.query, "count": len(results), "results": results}


@app.post("/search/hybrid")
async def search_hybrid(req: HybridSearchRequest) -> Dict[str, Any]:
    results = hybrid_search(query_text=req.query, metadata_filters=req.metadata_filters, top_k=req.top_k)
    return {"query": req.query, "count": len(results), "results": results}


@app.post("/rag")
async def rag(req: RAGRequest) -> Dict[str, Any]:
    rag_tool.top_k = req.top_k
    return rag_tool.search(req.query, metadata_filters=req.metadata_filters)


@app.post("/agent/query")
async def agent_query(req: AgentQueryRequest) -> Dict[str, Any]:
    global agent_instance
    if agent_instance is None or agent_instance.llm is None:
        try:
            agent_instance = SatelliteAgent(model_name=req.model_name)
        except Exception:
            agent_instance = None
    
    if agent_instance and agent_instance.llm:
        result = agent_instance.query(req.question)
        return result
    else:
        # Fallback: RAG 검색 결과를 간단 요약
        search = rag_tool.search(req.question, top_k=5)
        snippets: List[str] = []
        for i, r in enumerate(search["results"], 1):
            snippets.append(f"{i}. {r['caption']}")
        answer = "\n".join(snippets[:5]) if snippets else "관련 이미지를 찾지 못했습니다."
        return {
            "question": req.question,
            "answer": answer,
            "search_results": search["results"],
            "num_results": len(search["results"]),
            "note": "OPENAI_API_KEY 미설정 또는 오류로 RAG 요약으로 대체"
        }


class ChatRequest(BaseModel):
    message: str
    top_k: int = 24


@app.post("/chat")
async def chat(req: ChatRequest) -> JSONResponse:
    # RAG 우선 검색 → 에이전트 답변(키가 있으면) 생성
    rag_tool.top_k = req.top_k
    search = rag_tool.search(req.message, top_k=req.top_k)

    # 이미지 파일 경로를 웹 경로(/images/...)로 변환
    results_with_urls = []
    for r in search["results"]:
        abs_path = Path(r["image_path"]).resolve()
        try:
            rel = abs_path.relative_to(images_dir.resolve())
            url = f"/images/{rel.as_posix()}"
        except Exception:
            # 처리 디렉토리 하위가 아니면 직접 파일 URL 구성 불가 → 원본 경로 반환
            url = f"/images/{abs_path.name}"
        item = dict(r)
        item["image_url"] = url
        results_with_urls.append(item)

    # 에이전트 답변 시도
    global agent_instance
    answer = None
    if agent_instance is None:
        try:
            agent_instance = SatelliteAgent()
        except Exception:
            agent_instance = None
    if agent_instance and agent_instance.llm:
        try:
            # 서버에서 이미 검색한 결과를 에이전트에 전달
            resp = agent_instance.query(req.message, search_results=search["results"])
            answer = resp.get("answer")
        except Exception:
            answer = None

    if not answer:
        # RAG 결과 요약으로 대체
        if results_with_urls:
            answer = f"I found {len(results_with_urls)} relevant images. Please check the results below."
        else:
            answer = "관련 이미지를 찾지 못했습니다."

    return JSONResponse({
        "message": req.message,
        "answer": answer,
        "results": results_with_urls,
        "count": len(results_with_urls)
    })
