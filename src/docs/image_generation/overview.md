# Image Generation 기능 개요

## 목적
ComfyUI HTTP/WebSocket API를 통해 각 장면에 맞는 이미지를 자동 생성합니다. Z-Image Turbo 모델을 사용하여 고품질 애니메 일러스트를 생성합니다.

## 스코프
- ComfyUI 서버와 HTTP/WebSocket 통신
- 워크플로우 JSON 파일 로드 및 변환
- 프롬프트 업데이트 및 시드 랜덤화
- 이미지 생성 완료 대기
- 생성된 이미지 다운로드

## 입력
- `scenes: List[Dict[str, str]]` - 장면 리스트
  - 각 장면은 `image_prompt: str` 필드를 포함해야 함

## 출력
- `List[Dict[str, str]]` - 이미지 경로가 업데이트된 장면 리스트
  - 각 장면에 `image_path: str` 필드 추가/업데이트

## 의존성
- `requests` - HTTP API 호출
- `websocket-client` - WebSocket 통신
- `src.config.Config` - ComfyUI 설정
- ComfyUI 서버 (로컬 실행 필요)

