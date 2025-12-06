# TTS 처리 흐름

## 전체 흐름

1. **상태 확인**
   - `state`에서 `scenes` 추출
   - `scenes`가 없거나 비어있으면 `ValueError` 발생

2. **비동기 작업 준비**
   - 각 장면에 대해 `generate_audio_async()` 태스크 생성

3. **병렬 오디오 생성**
   - `asyncio.gather(*tasks)`로 모든 태스크 병렬 실행

4. **파일 생성 확인 및 상태 업데이트**
   - 각 장면에 `audio_path` 필드 추가/업데이트

