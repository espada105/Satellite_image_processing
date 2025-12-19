# 파트 3: AI 에이전트 (LangGraph)

## 📋 개요

이 파트는 **LangGraph 기반 AI 에이전트**로, 사용자의 질문에 대해 RAG 검색 결과를 바탕으로 자연어 답변을 생성하는 시스템을 다룹니다.

**핵심 목표**: 검색 결과 → LLM 컨텍스트 구성 → 자연어 답변 생성

---

## 🔄 전체 워크플로우

```
사용자 질문: "flooded area near river"
    ↓
[1단계] RAG 검색 (파트 2 활용)
    ↓
[2단계] 검색 결과를 컨텍스트로 구성
    ↓
[3단계] 시스템 프롬프트 + 컨텍스트 → LLM
    ↓
[4단계] 자연어 답변 생성
    ↓
최종 답변: "I found 8 images of flooded areas. These show..."
```

---

## 1단계: LangGraph 아키텍처

### 1.1 LangGraph 선택 이유

**LangChain vs LangGraph**:
- **LangChain**: 순차적 체인 구조
- **LangGraph**: 상태 기반 그래프 구조, 더 유연한 워크플로우

**선택 이유**:
1. **상태 관리**: `AgentState`로 검색 결과, 메시지 등을 안전하게 전달
2. **노드 기반**: 검색 노드, 답변 생성 노드로 명확한 분리
3. **확장성**: 향후 노드 추가/수정이 쉬움
4. **디버깅**: 각 노드의 입출력을 명확히 추적 가능

### 1.2 상태 정의 (AgentState)

```python
class AgentState(TypedDict):
    """에이전트 상태"""
    messages: Annotated[list, "대화 메시지"]
    query: str
    search_results: list
    final_answer: str
```

**각 필드의 역할**:
- `messages`: 대화 히스토리 (HumanMessage, AIMessage)
- `query`: 사용자 질문
- `search_results`: RAG 검색 결과 리스트
- `final_answer`: 최종 생성된 답변

**TypedDict 사용 이유**:
- 타입 안정성
- IDE 자동완성 지원
- LangGraph와 호환

### 1.3 그래프 구조

```python
def _build_graph(self) -> StateGraph:
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("search", self._search_node)
    workflow.add_node("generate_answer", self._generate_answer_node)
    
    # 엣지 추가
    workflow.set_entry_point("search")
    workflow.add_edge("search", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()
```

**그래프 흐름**:
```
[시작] → [search 노드] → [generate_answer 노드] → [종료]
```

**노드 설명**:
1. **search 노드**: RAG 검색 수행
2. **generate_answer 노드**: LLM으로 답변 생성

---

## 2단계: 검색 노드 (Search Node)

### 2.1 구현 (`_search_node`)

```python
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
    
    # RAG 검색 수행
    search_results = self.rag_tool.search(query)
    
    state["search_results"] = search_results["results"]
    state["query"] = query
    
    return state
```

**처리 과정**:
1. 상태에서 쿼리 추출 (없으면 메시지에서 추출)
2. RAG 도구로 검색 수행
3. 검색 결과를 상태에 저장
4. 상태 반환

### 2.2 최적화: 검색 결과 직접 전달

**문제 발견**:
- 서버(`server.py`)에서 이미 검색을 수행함 (`top_k=24`)
- 에이전트 내부에서 다시 검색하면 `top_k=5`로 제한됨
- **결과**: LLM이 "5개 이미지를 찾았다"고 말하지만 실제로는 24개가 표시됨

**해결 방법**:
- `query` 메서드에 `search_results` 파라미터 추가
- 검색 결과가 제공되면 검색 노드를 건너뛰고 답변 생성만 수행

```python
def query(self, question: str, search_results: List[Dict] = None) -> dict:
    if search_results is not None:
        # 검색 노드 건너뛰고 답변 생성만 수행
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "query": question,
            "search_results": search_results,
            "final_answer": ""
        }
        final_state = self._generate_answer_node(initial_state)
    else:
        # 그래프 실행 (검색 + 답변 생성)
        final_state = self.graph.invoke(initial_state)
```

**효과**:
- 검색 결과 개수 일치 (24개)
- 중복 검색 방지 (성능 향상)
- 일관성 확보

---

## 3단계: 답변 생성 노드 (Generate Answer Node)

### 3.1 시스템 프롬프트 설계

**초기 프롬프트 문제**:
- 개별 이미지를 나열하는 방식
- "I found 5 images: 1. Image ID: ... 2. Image ID: ..." 같은 형식
- 사용자 경험 저하 (이미지가 UI에 표시되므로 중복)

**최종 프롬프트**:
```python
system_prompt = """You are a helpful assistant specialized in analyzing satellite imagery.
You have access to a database of satellite images with captions.
Your task is to answer user questions based on the retrieved images and their captions.

When answering:
1. Be concise but informative (2-3 sentences)
2. Use the EXACT total number of retrieved images provided in the context
3. Summarize the key features or characteristics found in the images (e.g., locations, types of features, conditions)
4. Make the answer natural and engaging, not just a count
5. Do NOT list individual image details - the images are displayed below

IMPORTANT: 
- Always use the "Total retrieved images" number when stating how many images were found
- Provide a brief summary of what the images show (e.g., "showing flooded areas with houses and trees" or "depicting various road types in rural and urban settings")
- Make each answer slightly different even for similar queries to avoid repetition

Example good answers:
- "I found 8 images of flooded areas. These show various flooded regions with houses, trees, and water bodies, including both rural and coastal areas."
- "I found 12 images featuring roads and highways. The images depict different road types including country roads, highways, and urban streets in various settings."
- "I found 5 images showing golf courses. These aerial views capture golf courses surrounded by green fields, trees, and sometimes water features."
"""
```

**핵심 지침**:
1. **정확한 개수 사용**: 컨텍스트의 "Total retrieved images" 값 사용
2. **요약 중심**: 개별 이미지 나열 금지, 집합적 설명
3. **자연스러운 답변**: 단순 개수 나열이 아닌 의미 있는 설명
4. **반복 방지**: 유사한 쿼리에도 약간 다른 답변 생성

### 3.2 컨텍스트 구성

```python
# 컨텍스트 구성
total_images = len(search_results)
context_parts = [f"User question: {query}\n\n"]
context_parts.append(f"Total retrieved images: {total_images}\n\n")
context_parts.append("Sample retrieved images (showing top 5 for context):\n")

# 주요 특징 추출을 위해 더 많은 샘플 확인
for i, result in enumerate(search_results[:min(8, len(search_results))], 1):
    context_parts.append(f"\n{i}. Image ID: {result['image_id']}")
    context_parts.append(f"   Caption: {result['caption']}")
    if i <= 5:
        context_parts.append(f"   Similarity: {result['similarity']:.4f}")

context = "".join(context_parts)
```

**컨텍스트 구성 전략**:
1. **총 개수 명시**: "Total retrieved images: 24"
2. **샘플 제공**: 상위 8개 이미지의 캡션 제공 (LLM이 패턴 파악)
3. **유사도 점수**: 상위 5개만 유사도 점수 포함 (컨텍스트 길이 제한)

**왜 8개 샘플인가?**
- 5개만 제공하면 패턴 파악이 어려움
- 8개면 충분한 패턴 파악 가능
- 더 많으면 컨텍스트 길이 초과 (토큰 제한)

### 3.3 LLM 호출

```python
prompt = f"""{system_prompt}

{context}

Please provide a natural, informative answer to the user's question. 
- Use the exact total number ({total_images}) when mentioning how many images were found
- Summarize the key characteristics, features, or patterns you observe across the retrieved images
- Make the answer engaging and slightly different from previous answers if the query is similar
- Focus on what the images collectively show, not individual details"""

# LLM 호출
response = self.llm.invoke([HumanMessage(content=prompt)])
answer = response.content
```

**LLM 모델**: `gpt-4o-mini`
- **선택 이유**: 비용 효율적, 충분한 성능
- **Temperature**: 0.7 (다양성과 일관성 균형)

**에러 처리**:
- LLM 호출 실패 시 에러 메시지 반환
- OPENAI_API_KEY 미설정 시 graceful degradation

---

## 4단계: 프롬프트 엔지니어링 개선 과정

### 4.1 초기 문제점

**문제 1: 개수 불일치**
- LLM이 "5개 이미지를 찾았다"고 말하지만 실제로는 24개 표시
- 원인: 컨텍스트에 샘플 5개만 제공, 총 개수 미명시

**문제 2: 개별 나열**
- "1. Image ID: ... 2. Image ID: ..." 형식
- 사용자 경험 저하 (이미지가 UI에 표시되므로 중복)

**문제 3: 반복 답변**
- 유사한 쿼리에 동일한 답변 반복
- 다양성 부족

### 4.2 개선 과정

**1단계: 총 개수 명시**
```python
context_parts.append(f"Total retrieved images: {total_images}\n\n")
```
- 컨텍스트에 명확히 총 개수 포함
- 프롬프트에서 "EXACT total number" 사용 강조

**2단계: 요약 중심으로 변경**
- 개별 이미지 나열 금지
- 집합적 설명 요구

**3단계: 반복 방지 지침 추가**
- "Make each answer slightly different even for similar queries"
- Temperature 조정 (0.7)

### 4.3 최종 결과

**개선 전**:
> "I found 5 images featuring roads and highways. Here are the details: 1. **Image ID: ...** - **Caption**: ... 2. **Image ID: ...** - **Caption**: ..."

**개선 후**:
> "I found 24 images featuring roads and highways. The images depict different road types including country roads, highways, and urban streets in various settings."

**효과**:
- 개수 정확도: 100% (항상 정확한 개수 사용)
- 답변 품질: 자연스럽고 요약 중심
- 사용자 경험: 개선

---

## 5단계: LLM 설정 및 폴백

### 5.1 LLM 초기화

```python
def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY가 설정되지 않았습니다.")
        self.llm = None
    else:
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=OPENAI_API_KEY
        )
```

**에러 처리**:
- API 키 미설정 시 `self.llm = None`
- 답변 생성 노드에서 체크하여 폴백 메시지 반환

### 5.2 Graceful Degradation

**서버 측 폴백** (`server.py`):
```python
if not answer:
    # RAG 결과 요약으로 대체
    if results_with_urls:
        answer = f"I found {len(results_with_urls)} relevant images. Please check the results below."
    else:
        answer = "관련 이미지를 찾지 못했습니다."
```

**효과**:
- LLM이 없어도 기본적인 답변 제공
- 사용자 경험 유지

---

## 🎯 파트 3 핵심 성과

### 기술적 성과

1. **LangGraph 기반 에이전트 구축**
   - 상태 기반 그래프 구조
   - 검색 → 답변 생성 워크플로우

2. **프롬프트 엔지니어링**
   - 위성 이미지 분석 특화 프롬프트
   - 개수 정확도, 요약 중심 답변

3. **최적화**
   - 검색 결과 직접 전달 (중복 검색 방지)
   - 컨텍스트 구성 최적화

### 답변 품질

- **정확도**: 검색 결과 개수 100% 정확
- **자연스러움**: 요약 중심의 자연스러운 답변
- **일관성**: 유사한 쿼리에도 약간 다른 답변 (반복 방지)

---

## 🔍 면접에서 강조할 포인트

### 1. LangGraph 선택 이유

**질문**: "왜 LangGraph를 선택했나요?"

**답변**:
> "LangGraph를 선택한 이유는 상태 기반 그래프 구조로 워크플로우를 명확하게 정의할 수 있기 때문입니다. 검색 노드와 답변 생성 노드로 분리하여 각 단계를 독립적으로 관리할 수 있고, 향후 노드 추가나 수정이 쉬운 구조입니다. 또한 상태 관리가 안전하게 이루어져 디버깅과 모니터링이 용이했습니다."

### 2. 프롬프트 엔지니어링

**질문**: "프롬프트를 어떻게 개선했나요?"

**답변**:
> "초기에는 개별 이미지를 나열하는 방식이었는데, 사용자 경험을 개선하기 위해 요약 중심의 답변으로 변경했습니다. 또한 검색된 이미지 개수를 정확히 사용하기 위해 컨텍스트에 'Total retrieved images: N'을 명시하고, 프롬프트에서 'EXACT total number'를 사용하도록 강하게 지시했습니다. 이를 통해 개수 정확도를 100% 달성했습니다."

### 3. 검색 결과 직접 전달 최적화

**질문**: "에이전트에서 검색을 두 번 하지 않도록 최적화한 이유는?"

**답변**:
> "서버에서 이미 검색을 수행했는데, 에이전트 내부에서 다시 검색하면 top_k가 달라져서 개수 불일치가 발생했습니다. 또한 중복 검색은 성능 저하를 야기하므로, 검색 결과를 직접 전달하는 방식으로 최적화했습니다. 이를 통해 검색 결과 개수 일치와 성능 향상을 동시에 달성했습니다."

---

## 📚 관련 파일

- `scripts/agent/satellite_agent.py`: AI 에이전트 구현
- `scripts/rag/rag_tool.py`: RAG 도구 (검색 결과 제공)

---

## 🔄 다음 파트로의 연결

파트 3에서 생성한 답변은 **파트 4 (웹 서비스)**로 전달되어 사용자에게 표시됩니다. 웹 UI에서 검색 결과와 함께 자연어 답변을 보여줍니다.

