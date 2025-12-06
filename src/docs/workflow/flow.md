# Workflow 처리 흐름

## 전체 흐름 (메인 오케스트레이터)

```
[START]
  ↓
[script_pipeline 노드]
  ↓ (Script Pipeline 서브그래프 호출)
  ↓ (topic → scenes)
  ↓
[asset_pipeline 노드]
  ↓ (Asset Generation 서브그래프 호출)
  ↓ (scenes → scenes with assets)
  ↓
[video_pipeline 노드]
  ↓ (Video Assembly 서브그래프 호출)
  ↓ (scenes → final_video_path)
  ↓
[END]
```

## 상세 흐름

### 1. 메인 그래프 생성
- `create_video_graph()` 호출
- VideoState 기반 StateGraph 생성
- 3개 서브그래프 노드 추가 및 엣지 연결

### 2. Script Pipeline 실행
- `node_script_pipeline()` 호출
- Script Pipeline 서브그래프 빌드 및 실행
- 입력: `topic: str`
- 출력: `scenes: List[Dict[str, str]]` (script, image_prompt 포함)

### 3. Asset Generation 실행
- `node_asset_pipeline()` 호출
- Asset Generation 서브그래프 빌드 및 실행
- 입력: `scenes` (script, image_prompt 포함)
- 출력: `scenes` (audio_path, image_path, motion_frames_path 포함)

### 4. Video Assembly 실행
- `node_video_pipeline()` 호출
- Video Assembly 서브그래프 빌드 및 실행
- 입력: `scenes` (모든 에셋 경로 포함)
- 출력: `final_video_path: str`

## 서브그래프 내부 흐름

### Script Pipeline 서브그래프
```
[START] → script_writer → [END]
```

### Asset Generation 서브그래프
```
[START] → audio_generator → visual_generator → motion_generator → [END]
```

### Video Assembly 서브그래프
```
[START] → video_editor → [END]
```

## 상태 전이

```
초기 상태:
  topic: "AI의 미래"
  scenes: []
  final_video_path: ""

Script Pipeline 후:
  topic: "AI의 미래"
  scenes: [
    {
      "script": "...",
      "image_prompt": "...",
      "audio_path": "",
      "image_path": ""
    },
    ...
  ]
  final_video_path: ""

Asset Generation 후:
  topic: "AI의 미래"
  scenes: [
    {
      "script": "...",
      "image_prompt": "...",
      "audio_path": "output/topic/audio_scene_1.mp3",
      "image_path": "output/topic/image_scene_1.png",
      "motion_frames_path": [...]
    },
    ...
  ]
  final_video_path: ""

Video Assembly 후:
  topic: "AI의 미래"
  scenes: [...]
  final_video_path: "output/topic/final_output.mp4"
```

## 에러 처리

각 서브그래프는 독립적으로 에러를 처리합니다:
- 서브그래프 내부 노드의 재시도 로직
- 캐싱으로 인한 스킵
- 명확한 에러 메시지 전파

## 캐싱

각 서브그래프는 독립적으로 캐싱됩니다:
- Script Pipeline: `WorkflowCache.STAGE_SCRIPT`
- Asset Generation: `WorkflowCache.STAGE_AUDIO`, `STAGE_IMAGE`, `STAGE_MOTION`
- Video Assembly: `WorkflowCache.STAGE_VIDEO`
