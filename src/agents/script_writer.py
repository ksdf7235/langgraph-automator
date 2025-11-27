"""
대본 작성 에이전트
Ollama LLM을 사용하여 유튜브 쇼츠 대본을 생성합니다.
"""

import json
import logging
import requests
from typing import Dict, List

from src.config import Config
from src.utils.json_parser import extract_json_from_text

logger = logging.getLogger(__name__)


def call_ollama(prompt: str, model: str = None, temperature: float = None) -> str:
    """
    Ollama API를 직접 호출하여 LLM 응답을 받습니다.
    
    Args:
        prompt: 프롬프트 텍스트
        model: 모델 이름 (기본값: Config.OLLAMA_MODEL)
        temperature: 온도 설정 (기본값: Config.OLLAMA_TEMPERATURE)
        
    Returns:
        LLM 응답 텍스트
        
    Raises:
        requests.RequestException: API 호출 실패 시
        ValueError: 응답이 유효하지 않은 경우
    """
    model = model or Config.OLLAMA_MODEL
    temperature = temperature if temperature is not None else Config.OLLAMA_TEMPERATURE
    
    url = f"{Config.OLLAMA_URL}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    
    try:
        logger.info(f"Ollama API 호출: {model} (temperature: {temperature})")
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        
        if "response" not in result:
            raise ValueError(f"Ollama 응답에 'response' 키가 없습니다: {result}")
        
        content = result["response"]
        logger.debug(f"Ollama 응답 수신 (길이: {len(content)} 문자)")
        
        return content
        
    except requests.RequestException as e:
        logger.error(f"Ollama API 호출 실패: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Ollama 응답 JSON 파싱 실패: {e}")
        raise ValueError(f"Ollama 응답이 유효한 JSON이 아닙니다: {e}")


def create_script_prompt(topic: str) -> str:
    """
    대본 작성 프롬프트를 생성합니다.
    DeepSeek R1의 <think> 블록을 포함하도록 구성합니다.
    
    Args:
        topic: 비디오 주제
        
    Returns:
        완성된 프롬프트 텍스트
    """
    prompt = f"""<think>
사용자가 요청한 주제에 맞는 60초 이내 유튜브 쇼츠 대본을 작성해야 합니다.
주제: {topic}

요구사항을 분석하고, 각 장면의 대사와 이미지 프롬프트를 생성해야 합니다.
</think>

다음 주제에 맞는 60초 이내 유튜브 쇼츠 대본을 작성해주세요.

주제: {topic}

요구사항:
1. 3-5개의 장면으로 구성
2. 각 장면은 10-15초 분량의 대사(script)와 이미지 생성 프롬프트(image_prompt)를 포함
3. 대사는 자연스럽고 매력적이어야 함
4. 이미지 프롬프트는 영어로 작성하고, 구체적이고 시각적으로 표현

**중요**: 반드시 다음 JSON 형식으로만 출력하세요. 코드 블록 안에 JSON만 포함하세요:

```json
{{
  "scenes": [
    {{
      "script": "첫 번째 장면의 대사",
      "image_prompt": "A detailed image description in English"
    }},
    {{
      "script": "두 번째 장면의 대사",
      "image_prompt": "Another detailed image description in English"
    }}
  ]
}}
```

JSON 코드 블록 안의 JSON만 출력하고, 다른 설명이나 텍스트는 포함하지 마세요."""
    
    return prompt


def generate_script(topic: str) -> List[Dict[str, str]]:
    """
    주제에 맞는 쇼츠 대본을 생성합니다.
    
    Args:
        topic: 비디오 주제
        
    Returns:
        장면 리스트: [{"script": str, "image_prompt": str, "audio_path": "", "image_path": ""}, ...]
        
    Raises:
        ValueError: 대본 생성 또는 파싱 실패 시
    """
    logger.info(f"[대본 작성] 주제: {topic}")
    
    try:
        # 프롬프트 생성
        prompt = create_script_prompt(topic)
        
        # Ollama 호출
        response_text = call_ollama(prompt)
        
        logger.info(f"[대본 작성] LLM 응답 수신 (길이: {len(response_text)} 문자)")
        
        # JSON 추출 및 파싱
        parsed_data = extract_json_from_text(response_text)
        
        if "scenes" not in parsed_data:
            raise ValueError("응답에 'scenes' 키가 없습니다.")
        
        scenes = parsed_data["scenes"]
        
        if not isinstance(scenes, list) or len(scenes) == 0:
            raise ValueError(f"scenes가 유효한 리스트가 아닙니다: {scenes}")
        
        # 각 장면에 초기 경로 필드 추가
        for scene in scenes:
            if not isinstance(scene, dict):
                raise ValueError(f"장면이 유효한 딕셔너리가 아닙니다: {scene}")
            
            scene.setdefault("script", "")
            scene.setdefault("image_prompt", "")
            scene.setdefault("audio_path", "")
            scene.setdefault("image_path", "")
        
        logger.info(f"[대본 작성] 완료: {len(scenes)}개 장면 생성")
        
        return scenes
        
    except Exception as e:
        logger.error(f"[대본 작성] 오류 발생: {e}", exc_info=True)
        raise


def node_script_writer(state: Dict) -> Dict:
    """
    LangGraph 노드 함수: 대본 작성
    
    Args:
        state: 현재 상태 딕셔너리
        
    Returns:
        업데이트된 상태 딕셔너리
    """
    try:
        topic = state.get("topic", "")
        if not topic:
            raise ValueError("상태에 'topic'이 없습니다.")
        
        scenes = generate_script(topic)
        
        return {
            **state,
            "scenes": scenes
        }
        
    except Exception as e:
        logger.error(f"[대본 작성 노드] 오류: {e}", exc_info=True)
        raise

