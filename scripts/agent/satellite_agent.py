"""
위성 이미지 분석 AI 에이전트 (LangGraph)

사용자의 질문에 대해:
1. RAG 검색으로 관련 이미지 찾기
2. 이미지 분석 및 답변 생성
"""

import sys
import os
from typing import TypedDict, Annotated
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from scripts.rag.rag_tool import RAGTool
from scripts.utils.config import OPENAI_API_KEY


class AgentState(TypedDict):
    """에이전트 상태"""
    messages: Annotated[list, "대화 메시지"]
    query: str
    search_results: list
    final_answer: str


class SatelliteAgent:
    """위성 이미지 분석 에이전트"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Args:
            model_name: 사용할 LLM 모델 이름
            temperature: 모델 온도
        """
        if not OPENAI_API_KEY:
            print("⚠️  OPENAI_API_KEY가 설정되지 않았습니다.")
            print("   .env 파일에 OPENAI_API_KEY를 추가하거나 환경 변수로 설정하세요.")
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=OPENAI_API_KEY
            )
        
        self.rag_tool = RAGTool(top_k=5)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 그래프 구축"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("search", self._search_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        
        # 엣지 추가
        workflow.set_entry_point("search")
        workflow.add_edge("search", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def _search_node(self, state: AgentState) -> AgentState:
        """RAG 검색 노드"""
        query = state.get("query", "")
        
        if not query:
            # 마지막 사용자 메시지에서 쿼리 추출
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break
        
        print(f"🔍 검색 중: {query}")
        
        # RAG 검색 수행
        search_results = self.rag_tool.search(query)
        
        state["search_results"] = search_results["results"]
        state["query"] = query
        
        return state
    
    def _generate_answer_node(self, state: AgentState) -> AgentState:
        """답변 생성 노드"""
        if not self.llm:
            state["final_answer"] = "LLM이 설정되지 않았습니다. OPENAI_API_KEY를 확인하세요."
            return state
        
        query = state.get("query", "")
        search_results = state.get("search_results", [])
        messages = state.get("messages", [])
        
        # 시스템 프롬프트 구성
        system_prompt = """You are a helpful assistant specialized in analyzing satellite imagery.
You have access to a database of satellite images with captions.
Your task is to answer user questions based on the retrieved images and their captions.

When answering:
1. Be very concise (1-2 sentences maximum)
2. Just mention what you found (e.g., "I found X images with roads and highways")
3. Do NOT list all image details - the images are displayed below
4. If no relevant images are found, say so clearly

Example good answer: "I found 5 images featuring roads and highways in rural and urban settings."
Example bad answer: "I found several images. Image 1: ... Image 2: ..." (too detailed)
"""
        
        # 컨텍스트 구성
        context_parts = [f"User question: {query}\n\n"]
        context_parts.append("Retrieved images:\n")
        
        for i, result in enumerate(search_results[:5], 1):
            context_parts.append(f"\n{i}. Image ID: {result['image_id']}")
            context_parts.append(f"   Caption: {result['caption']}")
            context_parts.append(f"   Similarity: {result['similarity']:.4f}")
        
        context = "".join(context_parts)
        
        # 메시지 구성
        prompt = f"{system_prompt}\n\n{context}\n\nPlease answer the user's question based on the retrieved images."
        
        # LLM 호출
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content
        except Exception as e:
            answer = f"오류 발생: {str(e)}"
        
        state["final_answer"] = answer
        
        # 메시지에 추가
        new_messages = messages + [AIMessage(content=answer)]
        state["messages"] = new_messages
        
        return state
    
    def query(self, question: str) -> dict:
        """
        질문에 대한 답변 생성
        
        Args:
            question: 사용자 질문
        
        Returns:
            답변 딕셔너리
        """
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "query": question,
            "search_results": [],
            "final_answer": ""
        }
        
        # 그래프 실행
        final_state = self.graph.invoke(initial_state)
        
        return {
            "question": question,
            "answer": final_state["final_answer"],
            "search_results": final_state["search_results"],
            "num_results": len(final_state["search_results"])
        }


def create_agent(model_name: str = "gpt-4o-mini") -> SatelliteAgent:
    """
    에이전트 생성
    
    Args:
        model_name: LLM 모델 이름
    
    Returns:
        SatelliteAgent 인스턴스
    """
    return SatelliteAgent(model_name=model_name)


if __name__ == "__main__":
    # 테스트
    print("=" * 60)
    print("위성 이미지 분석 AI 에이전트 테스트")
    print("=" * 60)
    
    if not OPENAI_API_KEY:
        print("\n⚠️  OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 다음을 추가하세요:")
        print("   OPENAI_API_KEY=your_api_key_here")
        print("\n   또는 테스트 모드로 진행합니다...")
        
        # RAG 검색만 테스트
        from scripts.rag.rag_tool import RAGTool
        rag = RAGTool(top_k=3)
        
        test_queries = [
            "golf course images",
            "flooded areas",
            "buildings in cities"
        ]
        
        for query in test_queries:
            print(f"\n🔍 검색: {query}")
            results = rag.search(query)
            print(f"✅ 결과: {results['count']}개")
            for i, result in enumerate(results['results'], 1):
                print(f"  {i}. {result['image_id']}: {result['caption'][:50]}...")
    else:
        agent = create_agent()
        
        test_questions = [
            "골프장 이미지를 찾아줘",
            "홍수로 피해를 본 지역 이미지는?",
            "도시의 건물들이 보이는 이미지를 찾아줘"
        ]
        
        for question in test_questions:
            print(f"\n" + "=" * 60)
            print(f"질문: {question}")
            print("=" * 60)
            
            result = agent.query(question)
            
            print(f"\n답변:")
            print(result["answer"])
            print(f"\n검색된 이미지: {result['num_results']}개")

