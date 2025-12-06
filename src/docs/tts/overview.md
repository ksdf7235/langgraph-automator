# TTS (Text-to-Speech) 기능 개요

## 목적
edge-tts를 사용하여 텍스트를 한국어 음성 오디오로 변환합니다. 각 장면의 대사를 병렬로 처리하여 오디오 파일을 생성합니다.

## 스코프
- 텍스트를 한국어 음성으로 변환
- 여러 장면의 오디오를 병렬 생성
- 각 장면별 MP3 파일 생성
- 생성된 오디오 경로를 장면 데이터에 업데이트

## 입력
- `scenes: List[Dict[str, str]]` - 장면 리스트
  - 각 장면은 `script: str` 필드를 포함해야 함
- `output_dir: Path` - 출력 디렉토리

## 출력
- `List[Dict[str, str]]` - 오디오 경로가 업데이트된 장면 리스트
  - 각 장면에 `audio_path: str` 필드 추가/업데이트
  - 파일 형식: `audio_scene_{idx+1}.mp3`

## 의존성
- `edge_tts` - Microsoft Edge TTS 라이브러리
- `src.config.Config` - TTS 음성 설정 (`TTS_VOICE`)
- `asyncio` - 비동기 처리

## 품질 기준
- 모든 장면의 오디오가 성공적으로 생성되어야 함
- 생성된 오디오 파일이 실제로 존재해야 함
- 오디오 파일은 MP3 형식이어야 함
- 빈 스크립트는 오디오로 변환할 수 없음 (에러 발생)

