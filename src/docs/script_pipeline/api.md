# Script Pipeline API 명세

## Public 함수

### `build_script_graph() -> CompiledGraph[VideoState, VideoState]`

Script Pipeline 서브그래프를 생성하고 컴파일합니다.

**반환값:**
- 컴파일된 StateGraph 인스턴스

**그래프 구조:**
```
[START] → script_writer → [END]
```

---

### `node_script_writer(state: VideoState) -> VideoState`

LangGraph 노드 함수: 대본 작성 (기존 노드 그대로 사용)

**파라미터:**
- `state: VideoState` - 현재 상태 딕셔너리
  - 필수 키: `topic: str`

**반환값:**
- `VideoState` - 업데이트된 상태 딕셔너리
  - `scenes: List[Dict[str, str]]` - 생성된 장면 리스트

**에러:**
- `ValueError` - 상태에 'topic'이 없는 경우
- `requests.RequestException` - Ollama API 호출 실패 시
- `ValueError` - JSON 파싱 실패 시

**부가 효과:**
- 생성된 대본을 `output/{topic}/script.json`에 저장

## VideoState 필드

### 입력 필드 (필수)
- `topic: str` - 비디오 주제

### 출력 필드
- `scenes: List[Dict[str, str]]` - 장면 리스트
  - 각 장면:
    - `script: str` - 한국어 구어체 대사
    - `image_prompt: str` - 영어 이미지 프롬프트
    - `audio_path: str` - 빈 문자열
    - `image_path: str` - 빈 문자열

## 캐싱

- 캐시 단계: `WorkflowCache.STAGE_SCRIPT`
- 캐시 키: `scenes`
- 동일한 `topic`에 대해 캐시된 `scenes`가 있으면 스킵

## 재시도 정책

- 최대 재시도: 2회
- 재시도 지연: 2초 (지수 백오프)

