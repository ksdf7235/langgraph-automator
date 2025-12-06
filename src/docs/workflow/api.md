# Workflow API 명세

## Public 타입

### `VideoState` - 비디오 생성 워크플로우 상태

```python
class VideoState(TypedDict):
    topic: str                      # 비디오 주제
    scenes: List[Dict[str, str]]    # 장면 리스트
    final_video_path: str           # 최종 비디오 파일 경로
```

## Public 데코레이터

### `retry_node(max_retries: int = 2, delay: float = 1.0)`
노드 함수에 재시도 로직을 추가하는 데코레이터

### `with_cache(stage: str, result_keys: List[str])`
노드에 캐싱 로직을 추가하는 데코레이터

## Public 함수

### `create_video_graph(checkpoint: bool = False) -> StateGraph`

메인 오케스트레이터 그래프를 생성합니다.

**파라미터:**
- `checkpoint: bool` - 체크포인트 사용 여부 (기본값: False)

**반환값:**
- 컴파일된 StateGraph 인스턴스

**그래프 구조:**
```
[START] → script_pipeline → asset_pipeline → video_pipeline → [END]
```

---

### `create_video_graph_simple() -> StateGraph`

간단한 버전의 그래프 생성 (체크포인트 없음)

**반환값:**
- 컴파일된 StateGraph 인스턴스

---

## 서브그래프 노드 함수

### `node_script_pipeline(state: VideoState) -> VideoState`

Script Pipeline 서브그래프를 호출하는 노드

### `node_asset_pipeline(state: VideoState) -> VideoState`

Asset Generation 서브그래프를 호출하는 노드

### `node_video_pipeline(state: VideoState) -> VideoState`

Video Assembly 서브그래프를 호출하는 노드

## 서브그래프 빌더 함수

각 서브그래프 모듈에서 제공:

- `build_script_graph()` - Script Pipeline 서브그래프 생성
- `build_asset_graph()` - Asset Generation 서브그래프 생성
- `build_video_graph()` - Video Assembly 서브그래프 생성

## 사용 예시

```python
from src.workflow.graph import create_video_graph_simple, VideoState

# 그래프 생성
graph = create_video_graph_simple()

# 초기 상태 설정
initial_state: VideoState = {
    "topic": "AI의 미래",
    "scenes": [],
    "final_video_path": ""
}

# 그래프 실행
final_state = graph.invoke(initial_state)

# 결과 확인
print(f"최종 비디오: {final_state['final_video_path']}")
```

## 호환성

기존 인터페이스와 호환됩니다:
- `create_video_graph()` 함수 시그니처 유지
- `VideoState` 구조 유지
- 최종 결과물 형식 동일
