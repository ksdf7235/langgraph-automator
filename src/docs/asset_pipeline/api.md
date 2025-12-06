# Asset Generation API 명세

## Public 함수

### `build_asset_graph() -> CompiledGraph[VideoState, VideoState]`

Asset Generation 서브그래프를 생성하고 컴파일합니다.

**반환값:**
- 컴파일된 StateGraph 인스턴스

**그래프 구조:**
```
[START] → audio_generator → visual_generator → motion_generator → [END]
```

**참고:** motion_generator는 Config.MOTION_ENABLED가 True일 때만 실행됩니다.

---

## 내부 노드 함수

### `node_audio_generator(state: VideoState) -> VideoState`

오디오 생성 노드

**파라미터:**
- `state: VideoState` - 현재 상태 딕셔너리
  - 필수: `scenes` (각 장면에 `script` 필드 필요)

**반환값:**
- `VideoState` - 업데이트된 상태 딕셔너리
  - `scenes`: 각 장면에 `audio_path` 필드 추가/업데이트

**에러:**
- `ValueError` - scenes가 없거나 비어있는 경우
- `ValueError` - 장면에 script가 없는 경우
- `FileNotFoundError` - 오디오 파일이 생성되지 않은 경우

---

### `node_visual_generator(state: VideoState) -> VideoState`

이미지 생성 노드

**파라미터:**
- `state: VideoState` - 현재 상태 딕셔너리
  - 필수: `scenes` (각 장면에 `image_prompt` 필드 필요)

**반환값:**
- `VideoState` - 업데이트된 상태 딕셔너리
  - `scenes`: 각 장면에 `image_path` 필드 추가/업데이트

**에러:**
- `ValueError` - scenes가 없거나 비어있는 경우
- `ValueError` - 장면에 image_prompt가 없는 경우
- `RuntimeError` - ComfyUI 서버 연결 실패
- `requests.RequestException` - ComfyUI API 호출 실패

---

### `node_motion_generator(state: VideoState) -> VideoState`

모션 프레임 생성 노드

**파라미터:**
- `state: VideoState` - 현재 상태 딕셔너리
  - 필수: `scenes` (각 장면에 `image_path` 필드 필요)

**반환값:**
- `VideoState` - 업데이트된 상태 딕셔너리
  - `scenes`: 각 장면에 `motion_frames_path` 필드 추가/업데이트

**에러:**
- `ValueError` - scenes가 없거나 비어있는 경우
- `ValueError` - 장면에 image_path가 없는 경우
- `RuntimeError` - ComfyUI 서버 연결 실패
- `FileNotFoundError` - 이미지 파일이 없는 경우

**참고:**
- Config.MOTION_ENABLED가 False이면 스킵됩니다.
- 모션 생성 실패 시에도 에러를 발생시키지 않고 원본 scenes를 반환할 수 있습니다 (구현에 따라).

## VideoState 필드

### 입력 필드 (필수)
- `scenes: List[Dict[str, str]]` - 장면 리스트
  - 각 장면에 다음 필드 필요:
    - `script: str` (audio_generator용)
    - `image_prompt: str` (visual_generator용)
    - `image_path: str` (motion_generator용)

### 출력 필드
- `scenes: List[Dict[str, str]]` - 에셋 경로가 업데이트된 장면 리스트
  - 각 장면에 다음 필드 추가:
    - `audio_path: str`
    - `image_path: str`
    - `motion_frames_path: List[str]` (선택적)

## 캐싱

- 캐시 단계:
  - `WorkflowCache.STAGE_AUDIO` (audio_generator)
  - `WorkflowCache.STAGE_IMAGE` (visual_generator)
  - `WorkflowCache.STAGE_MOTION` (motion_generator)
- 각 노드는 독립적으로 캐싱됩니다.
- 캐시된 에셋 파일이 존재하지 않으면 캐시가 무효화됩니다.

## 재시도 정책

- audio_generator: 최대 2회, 1초 지연
- visual_generator: 최대 2회, 3초 지연
- motion_generator: 최대 2회, 3초 지연

