# Image Generation API 명세

## Public 클래스

### `ComfyUIClient`

ComfyUI HTTP/WebSocket API 클라이언트

**주요 메서드:**
- `check_health()` - 서버 헬스체크
- `load_workflow()` - 워크플로우 로드
- `update_prompt()` - 프롬프트 업데이트
- `execute_prompt()` - 프롬프트 실행
- `wait_for_completion()` - 완료 대기
- `download_result()` - 이미지 다운로드

---

## Public 함수

### `generate_images(scenes: List[Dict], output_dir: Path, retry_count: int = None) -> List[Dict]`

모든 장면의 이미지를 생성합니다.

