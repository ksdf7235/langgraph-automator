# TTS API 명세

## Public 함수

### `generate_audio_async(script: str, output_path: str, voice: str = None) -> None`

edge-tts를 사용하여 텍스트를 오디오로 변환합니다 (비동기).

**파라미터:**
- `script: str` - 변환할 텍스트
- `output_path: str` - 저장할 오디오 파일 경로
- `voice: str` - 사용할 음성 (기본값: `Config.TTS_VOICE`)

**반환값:**
- `None` - 반환값 없음 (파일로 저장)

**에러:**
- `ValueError` - 빈 스크립트인 경우
- `FileNotFoundError` - 오디오 파일이 생성되지 않은 경우
- `Exception` - edge-tts 호출 실패 시

---

### `generate_all_audios(scenes: List[Dict], output_dir: Path) -> List[Dict]`

모든 장면의 오디오를 생성합니다.

**파라미터:**
- `scenes: List[Dict]` - 장면 리스트
- `output_dir: Path` - 출력 디렉토리

**반환값:**
- `List[Dict]` - 오디오 경로가 업데이트된 장면 리스트

---

### `node_audio_generator(state: Dict) -> Dict`

LangGraph 노드 함수: 오디오 생성

