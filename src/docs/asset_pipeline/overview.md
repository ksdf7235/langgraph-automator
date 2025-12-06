# Asset Generation 서브그래프 개요

## 목적
Asset Generation 서브그래프는 대본(scenes) 정보를 받아 실제 에셋(오디오, 이미지, 모션)을 생성합니다. 이 서브그래프는 여러 에셋 생성 작업을 순차적으로 수행하여 각 장면에 필요한 모든 미디어 파일을 생성합니다.

## 스코프
- TTS를 사용한 오디오 생성
- ComfyUI를 사용한 이미지 생성
- Wan2.2 I2V를 사용한 모션 프레임 생성
- 생성된 에셋 경로를 scenes에 업데이트

## 입력
- `scenes: List[Dict[str, str]]` - 장면 리스트
  - 각 장면은 다음 필드를 포함해야 함:
    - `script: str` - 오디오 생성용 대사
    - `image_prompt: str` - 이미지 생성용 프롬프트

## 출력
- `scenes: List[Dict[str, str]]` - 에셋 경로가 업데이트된 장면 리스트
  - 각 장면에 다음 필드 추가/업데이트:
    - `audio_path: str` - 생성된 오디오 파일 경로
    - `image_path: str` - 생성된 이미지 파일 경로
    - `motion_frames_path: List[str]` - 생성된 모션 프레임 파일 경로 리스트 (모션 활성화 시)

## 책임 경계
- **이 서브그래프가 담당하는 것:**
  - 오디오 파일 생성 (TTS)
  - 이미지 파일 생성 (ComfyUI)
  - 모션 프레임 생성 (Wan2.2 I2V, 선택적)
  
- **이 서브그래프가 담당하지 않는 것:**
  - 대본 생성 (Script Pipeline 서브그래프)
  - 비디오 편집 (Video Assembly 서브그래프)

## 내부 노드

### audio_generator
- 역할: edge-tts를 사용하여 각 장면의 대사를 오디오로 변환
- 입력: `scenes` (script 필드 필요)
- 출력: `scenes` (audio_path 필드 업데이트)
- 캐시: `WorkflowCache.STAGE_AUDIO`
- 재시도: 2회, 1초 지연
- 특징: 병렬 처리 (asyncio)

### visual_generator
- 역할: ComfyUI를 사용하여 각 장면의 이미지 프롬프트로 이미지 생성
- 입력: `scenes` (image_prompt 필드 필요)
- 출력: `scenes` (image_path 필드 업데이트)
- 캐시: `WorkflowCache.STAGE_IMAGE`
- 재시도: 2회, 3초 지연
- 특징: 순차 처리 (ComfyUI API 제약)

### motion_generator
- 역할: Wan2.2 I2V를 사용하여 정적 이미지를 모션 프레임으로 변환
- 입력: `scenes` (image_path 필드 필요)
- 출력: `scenes` (motion_frames_path 필드 업데이트)
- 캐시: `WorkflowCache.STAGE_MOTION`
- 재시도: 2회, 3초 지연
- 특징: 선택적 (Config.MOTION_ENABLED로 제어)

## 실행 순서
1. audio_generator (오디오 생성)
2. visual_generator (이미지 생성)
3. motion_generator (모션 프레임 생성, 선택적)

이 순서는 의존성을 반영합니다:
- 오디오와 이미지는 독립적으로 생성 가능 (병렬 가능하지만 현재는 순차)
- 모션 프레임은 이미지가 먼저 생성되어야 함

## 향후 확장 포인트
- 오디오와 이미지 생성 병렬화
- 에셋 생성 실패 시 부분 재생성 로직
- 에셋 품질 검증 노드 추가

