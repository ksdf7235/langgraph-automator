"""
대본 작성 에이전트
Ollama LLM을 사용하여 유튜브 쇼츠 대본을 생성합니다.
"""

import json
import logging
import os
import requests
from pathlib import Path
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
        # DeepSeek R1 14B는 응답 생성에 시간이 오래 걸릴 수 있으므로 타임아웃을 600초(10분)로 증가
        response = requests.post(url, json=payload, timeout=600)
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
    - DeepSeek R1의 <think> 블록을 활용해 구조/길이를 먼저 설계하게 합니다.
    - Shorts용 훅/본문/결론 구조를 강하게 요구합니다.
    - JSON만 출력하도록 강제합니다.
    """
    prompt = f"""<think>
        당신은 유튜브 쇼츠 대본을 전문으로 쓰는 한국인 작가입니다.
        사용자가 요청한 주제에 맞는 30~45초 길이의 유튜브 쇼츠 대본을 작성해야 합니다.

        주제: {topic}

        먼저 다음을 머릿속으로 설계해야 합니다:
        1) 훅 장면 (scene 1)
        - 3~5초 분량
        - 시청자가 바로 멈추고 보게 만드는 한두 문장
        - 놀라운 사실, 도발적인 질문, 공감되는 한 마디 중 하나를 선택

        2) 본문 장면 (scene 2~3)
        - 각 장면 10~15초 분량
        - 핵심 포인트 1~2개만 선택해서 간결하게 설명
        - 너무 많은 정보를 넣지 말 것
        - 각 장면은 1~3문장 정도로 유지

        3) 마무리/CTA 장면 (마지막 scene)
        - 5~8초 분량
        - "그래서 뭐가 중요한지"를 한 문장으로 정리
        - 시청자가 다음 행동을 떠올리게 만드는 한 문장
            예: 앞으로의 변화, 준비해야 할 것, 생각해 볼 질문 등

        전체 길이는 30~45초를 목표로 하고,
        각 장면의 대사는 1~3문장, 한 문장은 25자 이내의 짧은 구어체 문장으로 구성하는 것이 좋습니다.

        이제 위 구조를 기준으로 장면 수(3~4개)를 결정하고,
        각 장면의 역할(훅 / 본문 / 마무리)을 먼저 정리한 뒤
        대사(script)와 이미지 프롬프트(image_prompt)를 설계하세요.
        </think>

        다음 주제에 맞는 30~45초 유튜브 쇼츠 대본을 작성해주세요.

        주제: {topic}

        **대본 작성 요구사항 (매우 중요):**
        1. 3~4개의 장면으로 구성:
        - scene 1: 훅 (강렬한 오프닝)
        - scene 2~3: 본문 (핵심 내용)
        - 마지막 scene: 정리/결론/CTA
        2. 대사(script)는 **반드시 자연스러운 한국어 구어체**로 작성해야 합니다:
        - 일상 대화처럼 자연스럽고 부드러운 표현 사용
        - "~해요", "~입니다", "~네요", "~거예요" 같은 자연스러운 종결어미 사용
        - 너무 문어체이거나 설명충 느낌의 긴 문장 금지
        - 각 장면은 1~3문장, 한 문장은 25자 이내로 짧게 유지
        - 감정과 리액션이 자연스럽게 드러나는 표현을 섞기
        - 시청자가 "어, 내 얘긴데?"라고 느낄 수 있는 공감형 톤

        예시 (좋음):
        - "요즘 AI 얘기 너무 많이 나오죠? 솔직히 좀 불안하지 않아요?"
        - "근데 진짜 중요한 건, 우리 일이 완전히 사라지는 게 아니라는 거예요."

        예시 (나쁨):
        - "인공지능은 현대 사회에서 점차 중요한 역할을 수행하고 있습니다." (딱딱한 문어체)

        3. 이미지 프롬프트(image_prompt)는 **최고 품질의 애니메 일러스트**를 생성하도록 작성:
        - 영어로 작성
        - **반드시 다음 고품질 태그로 시작:**
            "masterpiece, best quality, ultra detailed, anime style, consistent art style, JANKU style, detailed anime illustration, "
        - 전체 영상에서 스타일이 일관되도록, 공통 분위기를 유지:
            - 예: "futuristic city, neon lights, blue and purple color palette" 등
        - 캐릭터가 등장하는 경우:
            - "cute anime character, consistent character design, beautiful detailed face, expressive eyes, "
            - 같은 인물처럼 보이도록 머리색, 헤어스타일, 대략적인 나이대/분위기를 유지
        - 각 장면별로 구도/샷 타입 힌트를 포함:
            - scene 1: dynamic wide shot, impactful composition
            - scene 2~3: medium shot, focus on main character or key object
            - 마지막 scene: close-up, strong emphasis, maybe text overlay vibes

        **이미지 프롬프트 고품질 가이드:**
        - 모든 이미지 프롬프트는 반드시 다음으로 시작:
        "masterpiece, best quality, ultra detailed, anime style, consistent art style, JANKU style, detailed anime illustration, "
        - 캐릭터가 등장하는 경우 추가:
        "cute anime character, consistent character design, beautiful detailed face, expressive eyes, "
        - 그 뒤에 구체적인 장면 묘사를 영어로 적습니다:
        - 배경: 장소, 시간대, 분위기
        - 조명: warm / cool / neon / soft light 등
        - 색감: dominant colors
        - 샷 타입: wide shot, medium shot, close-up 등

        **중요**: 반드시 다음 JSON 형식으로만 출력하세요. 코드 블록 안에 JSON만 포함하세요:

        ```json
        {{
        "scenes": [
            {{
            "script": "완전히 자연스러운 한국어 구어체로 작성된 첫 번째 장면의 대사 (훅 역할)",
            "image_prompt": "masterpiece, best quality, ultra detailed, anime style, consistent art style, JANKU style, detailed anime illustration, [scene 1에 어울리는 강렬한 wide shot 장면 묘사]"
            }},
            {{
            "script": "완전히 자연스러운 한국어 구어체로 작성된 두 번째 장면의 대사 (본문 역할)",
            "image_prompt": "masterpiece, best quality, ultra detailed, anime style, consistent art style, JANKU style, detailed anime illustration, [scene 2에 어울리는 medium shot 장면 묘사]"
            }}
        ]
        }}
     "scenes" 배열의 길이는 3~4개여야 합니다.

     각 scene의 "script"와 "image_prompt"는 비어 있으면 안 됩니다.

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
        
        # 대본을 주제별 폴더에 JSON 파일로 저장
        output_dir = Config.get_output_dir(topic=topic)
        script_path = output_dir / "script.json"
        
        script_data = {
            "topic": topic,
            "scenes": scenes
        }
        
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                json.dump(script_data, f, ensure_ascii=False, indent=2)
            logger.info(f"[대본 작성] 대본 저장 완료: {script_path}")
        except Exception as e:
            logger.warning(f"[대본 작성] 대본 저장 실패: {e}")
        
        return {
            **state,
            "scenes": scenes
        }
        
    except Exception as e:
        logger.error(f"[대본 작성 노드] 오류: {e}", exc_info=True)
        raise

