# 워크플로우 리팩터링 설계 문서

## 목적
단일 StateGraph를 "3개 서브그래프 + 1개 메인 오케스트레이터" 구조로 개편하여, 코드 구조를 더 명확하게 하고 향후 확장성을 높입니다.

## 현재 구조 분석

### 현재 노드 흐름
```
script_writer → audio_generator → visual_generator → motion_generator → video_editor
```

### VideoState 구조
```python
class VideoState(TypedDict):
    topic: str                      # 입력: 비디오 주제
    scenes: List[Dict[str, str]]    # 각 장면의 데이터 (script, image_prompt, audio_path, image_path, motion_frames_path 등)
    final_video_path: str           # 출력: 최종 비디오 파일 경로
```

### 노드별 의존성 분석

1. **script_writer**
   - 입력: `topic`
   - 출력: `scenes` (script, image_prompt 포함, audio_path, image_path는 빈 문자열)
   - 책임: LLM을 사용하여 대본 생성

2. **audio_generator**
   - 입력: `scenes` (script 필드 필요)
   - 출력: `scenes` (audio_path 필드 업데이트)
   - 책임: TTS로 오디오 생성

3. **visual_generator**
   - 입력: `scenes` (image_prompt 필드 필요)
   - 출력: `scenes` (image_path 필드 업데이트)
   - 책임: ComfyUI로 이미지 생성

4. **motion_generator**
   - 입력: `scenes` (image_path 필드 필요)
   - 출력: `scenes` (motion_frames_path 필드 업데이트)
   - 책임: Wan2.2 I2V로 모션 프레임 생성

5. **video_editor**
   - 입력: `scenes` (audio_path, image_path 또는 motion_frames_path 필요)
   - 출력: `final_video_path`
   - 책임: 최종 비디오 편집 및 렌더링

## 목표 아키텍처

### 서브그래프 분리 전략

#### 1. Script Pipeline Graph
**역할:** 대본 및 씬 메타데이터 생성

**노드:**
- `script_writer`: 기존 노드 그대로 사용

**입력:**
- `topic: str`

**출력:**
- `scenes: List[Dict[str, str]]` (script, image_prompt 포함)

**책임:**
- 주제를 받아 대본 생성
- 각 장면별 텍스트 정보 생성
- 향후 확장: script_planner, script_refiner 추가 가능

---

#### 2. Asset Generation Graph
**역할:** 오디오, 이미지, 모션 등 실제 에셋 생성

**노드:**
- `audio_generator`: TTS로 오디오 생성
- `visual_generator`: ComfyUI로 이미지 생성
- `motion_generator`: Wan2.2 I2V로 모션 프레임 생성

**입력:**
- `scenes: List[Dict[str, str]]` (script, image_prompt 필요)

**출력:**
- `scenes: List[Dict[str, str]]` (audio_path, image_path, motion_frames_path 업데이트)

**책임:**
- 스크립트 정보를 바탕으로 모든 에셋 생성
- 향후 확장: 에셋 생성 병렬화 가능 (audio/visual 병렬)

---

#### 3. Video Assembly Graph
**역할:** 최종 비디오 편집 및 렌더링

**노드:**
- `video_editor`: 최종 비디오 생성

**입력:**
- `scenes: List[Dict[str, str]]` (audio_path, image_path 또는 motion_frames_path 필요)

**출력:**
- `final_video_path: str`

**책임:**
- 모든 에셋을 결합하여 최종 비디오 생성
- 향후 확장: 썸네일 생성, 자막 렌더링 등

---

#### 4. Main Orchestrator Graph
**역할:** 전체 파이프라인 오케스트레이션

**노드:**
- `script_pipeline`: Script Pipeline 서브그래프 호출
- `asset_pipeline`: Asset Generation 서브그래프 호출
- `video_pipeline`: Video Assembly 서브그래프 호출

**흐름:**
```
[START] → script_pipeline → asset_pipeline → video_pipeline → [END]
```

**책임:**
- 서브그래프들을 순차적으로 호출
- 글로벌 정책 관리 (재시도, 캐싱 등)
- 향후 확장: 조건 분기 (예: 대본 길이 초과 시 script_pipeline 재실행)

## 구현 전략

### LangGraph 서브그래프 패턴

LangGraph에서 서브그래프를 노드처럼 사용하는 방법:
1. 각 서브그래프를 독립적인 StateGraph로 구성
2. 컴파일된 그래프를 래퍼 함수로 감싸서 노드처럼 사용

예시:
```python
def build_script_graph() -> CompiledGraph:
    graph = StateGraph(VideoState)
    graph.add_node("script_writer", node_script_writer_with_retry)
    graph.set_entry_point("script_writer")
    graph.add_edge("script_writer", END)
    return graph.compile()

def node_script_pipeline(state: VideoState) -> VideoState:
    """Script Pipeline 서브그래프 래퍼"""
    script_graph = build_script_graph()
    result = script_graph.invoke(state)
    return result
```

### VideoState 유지

VideoState는 모든 서브그래프에서 공유되며, 각 서브그래프는 필요한 필드만 업데이트합니다.

### 캐시 및 재시도 정책

- 각 서브그래프 내부 노드에는 기존과 동일한 캐시/재시도 데코레이터 적용
- 메인 오케스트레이터 레벨에서는 서브그래프 전체 실패 시 재시도 가능 (향후 확장)

## 파일 구조

### 새로운 파일
- `src/workflow/script_graph.py` - Script Pipeline 서브그래프
- `src/workflow/asset_graph.py` - Asset Generation 서브그래프
- `src/workflow/video_graph.py` - Video Assembly 서브그래프
- `src/workflow/graph.py` - 메인 오케스트레이터 그래프 (기존 파일 수정)

### 문서 파일
- `src/docs/workflow/overview.md` - 업데이트
- `src/docs/workflow/flow.md` - 업데이트
- `src/docs/workflow/api.md` - 업데이트
- `src/docs/script_pipeline/` - 새 문서 폴더
- `src/docs/asset_pipeline/` - 새 문서 폴더
- `src/docs/video_pipeline/` - 새 문서 폴더

## 호환성 유지

### 기존 인터페이스 유지
- `create_video_graph()` 함수 시그니처 유지
- 입력/출력 형식 동일 유지
- 최종 결과물 (비디오 파일, 로그 등) 동일 유지

### 테스트 전략
- 기존 테스트가 계속 작동하도록 보장
- 리팩터링 전/후 동일한 입력으로 동일한 결과 생성 확인

## 단계별 작업 계획

1. ✅ [1단계] 현재 구조 파악 및 설계 문서 작성
2. ⏳ [2단계] Script Pipeline 서브그래프 분리
3. ⏳ [3단계] Asset Generation 서브그래프 분리
4. ⏳ [4단계] Video Assembly 서브그래프 분리
5. ⏳ [5단계] 메인 오케스트레이터 구현
6. ⏳ [6단계] 문서 업데이트 및 테스트

## 참고사항

- 모든 기능은 기존과 동일하게 작동해야 함
- 문서 우선 원칙: 문서 작성 → 코드 구현 순서
- 각 서브그래프는 독립적으로 테스트 가능해야 함

