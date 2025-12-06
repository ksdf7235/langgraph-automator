# Script Writer 처리 흐름

## 전체 흐름

1. **상태 확인**

   - `state`에서 `topic` 추출
   - `topic`이 없으면 `ValueError` 발생

2. **프롬프트 생성**

   - `create_script_prompt(topic)` 호출
   - DeepSeek R1용 프롬프트 생성
   - `<think>` 블록 포함
   - JSON 출력 형식 지시 포함

3. **LLM 호출**

   - `call_ollama(prompt)` 호출
   - Ollama API에 HTTP POST 요청
   - 타임아웃: 600초 (10분)
   - 응답에서 `response` 필드 추출

4. **JSON 추출 및 파싱**

   - `extract_json_from_text(response_text)` 호출
   - `<think>` 태그 제거
   - JSON 코드 블록 또는 JSON 객체 추출
   - `json.loads()`로 파싱

5. **데이터 검증**

   - `parsed_data`에 'scenes' 키 존재 확인
   - `scenes`가 유효한 리스트인지 확인
   - 각 장면이 유효한 딕셔너리인지 확인
   - 각 장면에 필수 필드 추가:
     - `script` (기본값: 빈 문자열)
     - `image_prompt` (기본값: 빈 문자열)
     - `audio_path` (기본값: 빈 문자열)
     - `image_path` (기본값: 빈 문자열)

6. **대본 저장**

   - `output/{topic}/script.json`에 저장
   - 저장 실패 시 경고만 출력 (프로세스는 계속)

7. **상태 업데이트**
   - `state`에 `scenes` 필드 추가/업데이트
   - 업데이트된 상태 반환

## 에러 처리

- **Ollama API 호출 실패**: `requests.RequestException` 발생
- **JSON 파싱 실패**: `ValueError` 발생
- **데이터 검증 실패**: `ValueError` 발생
- 모든 에러는 로깅 후 상위로 전파
