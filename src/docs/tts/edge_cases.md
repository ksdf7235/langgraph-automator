# TTS 엣지 케이스 및 Known Issues

## 엣지 케이스

### 1. 빈 스크립트
**증상:** `script` 필드가 비어있거나 공백만 포함

**대응:**
- `generate_audio_async()`에서 `script.strip()` 체크
- 빈 스크립트면 `ValueError` 발생

---

### 2. 오디오 파일 생성 실패
**증상:** `edge_tts.Communicate().save()` 호출 후 파일이 생성되지 않음

**대응:**
- 파일 존재 확인
- 파일이 없으면 `FileNotFoundError` 발생

