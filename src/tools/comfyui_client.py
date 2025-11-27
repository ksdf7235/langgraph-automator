"""
ComfyUI 클라이언트 모듈
ComfyUI API와 웹소켓을 사용하여 이미지를 생성합니다.
"""

import copy
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import requests
import websocket

from src.config import Config

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """ComfyUI API 클라이언트"""
    
    def __init__(self, base_url: str = None, ws_url: str = None, timeout: int = None):
        """
        ComfyUI 클라이언트 초기화
        
        Args:
            base_url: ComfyUI 서버 URL
            ws_url: ComfyUI 웹소켓 URL
            timeout: 타임아웃 (초)
        """
        self.base_url = base_url or Config.COMFYUI_URL
        self.ws_url = ws_url or Config.COMFYUI_WS_URL
        self.timeout = timeout or Config.COMFYUI_TIMEOUT
        self.client_id = str(uuid.uuid4())
        self.api_key = Config.COMFYUI_API_KEY if hasattr(Config, 'COMFYUI_API_KEY') else ""
        
    def get_available_models(self) -> List[str]:
        """
        ComfyUI에서 사용 가능한 체크포인트 모델 목록을 가져옵니다.
        
        Returns:
            사용 가능한 모델 파일명 리스트
            
        Raises:
            requests.RequestException: API 호출 실패 시
        """
        try:
            url = f"{self.base_url}/object_info"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            object_info = response.json()
            checkpoint_info = object_info.get("CheckpointLoaderSimple", {})
            input_info = checkpoint_info.get("input", {})
            required = input_info.get("required", {})
            ckpt_name_info = required.get("ckpt_name", [])
            
            if isinstance(ckpt_name_info, list) and len(ckpt_name_info) > 0:
                models = ckpt_name_info[0] if isinstance(ckpt_name_info[0], list) else ckpt_name_info
                logger.info(f"사용 가능한 모델 {len(models)}개 발견")
                return models
            else:
                logger.warning("사용 가능한 모델을 찾을 수 없습니다.")
                return []
                
        except requests.RequestException as e:
            logger.error(f"모델 목록 조회 실패: {e}")
            return []
    
    def load_workflow(self, workflow_path: str = None) -> Dict:
        """
        ComfyUI 워크플로우 JSON 파일을 로드하고 모델을 자동으로 설정합니다.
        
        Args:
            workflow_path: 워크플로우 JSON 파일 경로
            
        Returns:
            워크플로우 딕셔너리
        """
        workflow_path = workflow_path or Config.COMFYUI_WORKFLOW_PATH
        
        if not os.path.exists(workflow_path):
            logger.warning(f"워크플로우 파일이 없어 기본 워크플로우를 생성합니다: {workflow_path}")
            default_workflow = self._create_default_workflow()
            
            with open(workflow_path, 'w', encoding='utf-8') as f:
                json.dump(default_workflow, f, indent=2, ensure_ascii=False)
            
            workflow = default_workflow
        else:
            try:
                with open(workflow_path, 'r', encoding='utf-8') as f:
                    workflow = json.load(f)
                logger.debug(f"워크플로우 로드 완료: {workflow_path}")
            except Exception as e:
                logger.error(f"워크플로우 로드 실패: {e}")
                raise
        
        # CheckpointLoaderSimple 노드 찾아서 모델 자동 설정
        checkpoint_node_id = None
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == "CheckpointLoaderSimple":
                checkpoint_node_id = node_id
                break
        
        if checkpoint_node_id:
            # 환경 변수에서 지정된 모델이 있으면 우선 사용
            if Config.COMFYUI_CHECKPOINT:
                workflow[checkpoint_node_id]["inputs"]["ckpt_name"] = Config.COMFYUI_CHECKPOINT
                logger.info(f"환경 변수에서 지정된 모델 사용: {Config.COMFYUI_CHECKPOINT}")
            else:
                # 사용 가능한 모델 목록 가져오기
                available_models = self.get_available_models()
                
                if available_models:
                    # 첫 번째 모델 사용
                    selected_model = available_models[0]
                    workflow[checkpoint_node_id]["inputs"]["ckpt_name"] = selected_model
                    logger.info(f"모델 자동 선택: {selected_model}")
                else:
                    # 모델이 없으면 기본값 유지하되 경고
                    current_model = workflow[checkpoint_node_id]["inputs"].get("ckpt_name", "")
                    if current_model:
                        logger.warning(f"사용 가능한 모델을 찾을 수 없습니다. 기본 모델 사용: {current_model}")
                    else:
                        logger.error("사용 가능한 모델이 없고 기본 모델도 설정되지 않았습니다.")
        
        return workflow
    
    def _create_default_workflow(self) -> Dict:
        """기본 워크플로우 생성"""
        return {
            "1": {
                "inputs": {
                    "text": "default prompt",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Prompt)"}
            },
            "2": {
                "inputs": {
                    "text": "",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Negative)"}
            },
            "3": {
                "inputs": {
                    "width": 1024,
                    "height": 1024,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Empty Latent Image"}
            },
            "4": {
                "inputs": {
                    "ckpt_name": Config.COMFYUI_CHECKPOINT or "v1-5-pruned-emaonly.safetensors"
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint"}
            },
            "5": {
                "inputs": {
                    "seed": 12345,
                    "steps": 20,
                    "cfg": 7,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["1", 0],
                    "negative": ["2", 0],
                    "latent_image": ["3", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"}
            },
            "6": {
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"}
            },
            "7": {
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage",
                "_meta": {"title": "Save Image"}
            }
        }
    
    def update_prompt(self, workflow: Dict, prompt: str) -> Dict:
        """
        워크플로우에서 프롬프트 노드를 찾아 프롬프트를 업데이트합니다.
        딥카피를 사용하여 원본 워크플로우를 보호합니다.
        
        Args:
            workflow: ComfyUI 워크플로우 딕셔너리
            prompt: 새로운 프롬프트
            
        Returns:
            업데이트된 워크플로우 (딥카피)
        """
        if not prompt or not prompt.strip():
            raise ValueError("프롬프트가 비어있습니다.")
        
        # 딥카피 생성
        updated_workflow = copy.deepcopy(workflow)
        
        # CLIPTextEncode 노드 찾기
        # positive 프롬프트는 보통 빈 문자열이 아니거나 긴 텍스트를 가짐
        prompt_node_id = None
        
        for node_id, node_data in updated_workflow.items():
            if node_data.get("class_type") == "CLIPTextEncode":
                inputs = node_data.get("inputs", {})
                if "text" in inputs:
                    current_text = inputs.get("text", "")
                    # negative 프롬프트가 아닌 경우 (비어있지 않거나 긴 경우)
                    # 또는 _meta에 "Prompt"가 포함된 경우
                    meta_title = node_data.get("_meta", {}).get("title", "")
                    if "Prompt" in meta_title and "Negative" not in meta_title:
                        prompt_node_id = node_id
                        break
                    elif current_text and current_text != "" and len(current_text) > 10:
                        prompt_node_id = node_id
                        break
        
        if not prompt_node_id:
            # 첫 번째 CLIPTextEncode 노드를 사용
            for node_id, node_data in updated_workflow.items():
                if node_data.get("class_type") == "CLIPTextEncode":
                    prompt_node_id = node_id
                    break
        
        if not prompt_node_id:
            raise ValueError("워크플로우에서 CLIPTextEncode 노드를 찾을 수 없습니다.")
        
        updated_workflow[prompt_node_id]["inputs"]["text"] = prompt
        logger.debug(f"프롬프트 업데이트: 노드 {prompt_node_id} - {prompt[:50]}...")
        
        return updated_workflow
    
    def execute_prompt(self, workflow: Dict, retry_count: int = None) -> str:
        """
        ComfyUI에 프롬프트를 실행 요청합니다.
        
        Args:
            workflow: 실행할 워크플로우
            retry_count: 재시도 횟수 (기본값: Config.COMFYUI_RETRY_COUNT)
            
        Returns:
            프롬프트 ID
            
        Raises:
            requests.RequestException: API 호출 실패 시
        """
        retry_count = retry_count or Config.COMFYUI_RETRY_COUNT
        
        prompt_payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        
        url = f"{self.base_url}/prompt"
        
        # API 키가 있으면 헤더에 추가
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            logger.debug("ComfyUI API 키를 헤더에 추가했습니다.")
        
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                logger.debug(f"ComfyUI 프롬프트 실행 요청 (시도 {attempt + 1}/{retry_count + 1})")
                response = requests.post(url, json=prompt_payload, headers=headers, timeout=10)
                
                # 400 오류인 경우 상세 정보 로깅
                if response.status_code == 400:
                    try:
                        error_detail = response.json()
                        logger.error(f"ComfyUI 400 오류 상세: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                    except:
                        logger.error(f"ComfyUI 400 오류 응답 본문: {response.text[:500]}")
                
                response.raise_for_status()
                
                result = response.json()
                prompt_id = result.get("prompt_id")
                
                if not prompt_id:
                    raise ValueError("ComfyUI에서 prompt_id를 받지 못했습니다.")
                
                logger.info(f"프롬프트 ID: {prompt_id}")
                return prompt_id
                
            except requests.RequestException as e:
                last_error = e
                if attempt < retry_count:
                    wait_time = (attempt + 1) * 2  # 지수 백오프
                    logger.warning(f"ComfyUI API 호출 실패 (시도 {attempt + 1}/{retry_count + 1}), {wait_time}초 후 재시도: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"ComfyUI API 호출 최종 실패: {e}")
        
        raise last_error
    
    def _get_history_images(self, prompt_id: str, max_retries: int = 10, retry_delay: int = 3) -> List[str]:
        """
        ComfyUI API를 통해 히스토리에서 이미지 파일명을 조회합니다.
        이미지 생성 완료를 기다리기 위해 여러 번 재시도합니다.
        
        Args:
            prompt_id: 프롬프트 ID
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간 대기 시간 (초)
            
        Returns:
            생성된 이미지 파일명 리스트
        """
        logger.info(f"히스토리 API로 이미지 조회 시작 (프롬프트 ID: {prompt_id}, 최대 {max_retries}회 시도)")
        
        for attempt in range(max_retries):
            try:
                # ComfyUI 히스토리 API는 전체 히스토리를 반환
                url = f"{self.base_url}/history"
                logger.debug(f"히스토리 API 호출: {url} (시도 {attempt + 1}/{max_retries})")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                history = response.json()
                images = []
                
                # 히스토리 구조 확인
                if not isinstance(history, dict):
                    logger.warning(f"히스토리 응답이 딕셔너리가 아닙니다: {type(history)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return []
                
                # 히스토리에서 해당 prompt_id 찾기
                if prompt_id in history:
                    prompt_data = history[prompt_id]
                    outputs = prompt_data.get("outputs", {})
                    
                    logger.info(f"히스토리에서 프롬프트 ID {prompt_id} 발견, 출력 노드 수: {len(outputs)}")
                    
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            for img_info in node_output["images"]:
                                filename = img_info.get("filename")
                                if filename:
                                    images.append(filename)
                                    logger.info(f"히스토리에서 이미지 발견: {filename} (노드 {node_id})")
                
                if images:
                    logger.info(f"히스토리 API에서 {len(images)}개 이미지 발견 (시도 {attempt + 1}/{max_retries})")
                    return images
                else:
                    if prompt_id in history:
                        logger.warning(f"프롬프트 ID {prompt_id}는 히스토리에 있지만 이미지가 없습니다. 출력 구조: {list(prompt_data.keys())}")
                    else:
                        # 히스토리 키 목록 확인
                        history_keys = list(history.keys())
                        logger.info(f"히스토리에 {len(history_keys)}개 항목이 있습니다. 프롬프트 ID {prompt_id}를 찾는 중...")
                        if history_keys:
                            logger.debug(f"히스토리 키 목록 (최대 5개): {history_keys[:5]}")
                            # 유사한 ID 찾기
                            similar_ids = [k for k in history_keys if prompt_id[:8] in k or k[:8] in prompt_id]
                            if similar_ids:
                                logger.debug(f"유사한 프롬프트 ID 발견: {similar_ids[:3]}")
                    
                    if attempt < max_retries - 1:
                        logger.info(f"히스토리에서 이미지를 찾지 못했습니다. {retry_delay}초 후 재시도... (시도 {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"히스토리에서 프롬프트 ID {prompt_id}를 찾을 수 없거나 이미지가 없습니다 (최대 재시도 횟수 도달)")
                
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"히스토리 API 조회 실패, {retry_delay}초 후 재시도 (시도 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"히스토리 API 조회 최종 실패: {e}")
            except Exception as e:
                logger.error(f"히스토리 API 조회 중 예상치 못한 오류: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        logger.error(f"히스토리 API로 이미지를 찾지 못했습니다 (프롬프트 ID: {prompt_id})")
        return []
    
    def wait_for_completion(self, prompt_id: str, timeout: int = None, use_websocket: bool = False) -> List[str]:
        """
        ComfyUI 이미지 생성 완료를 대기합니다.
        기본적으로 히스토리 API를 사용하며, use_websocket=True 시 웹소켓을 먼저 시도합니다.
        
        Args:
            prompt_id: 프롬프트 ID
            timeout: 타임아웃 (초, 기본값: self.timeout)
            use_websocket: 웹소켓 사용 여부 (기본값: False)
            
        Returns:
            생성된 이미지 파일명 리스트
            
        Raises:
            TimeoutError: 타임아웃 발생 시
            ValueError: 이미지가 생성되지 않은 경우
        """
        timeout = timeout or self.timeout
        start_time = time.time()
        images = []
        
        # 웹소켓 사용 시 먼저 시도
        if use_websocket:
            images = self._wait_via_websocket(prompt_id, timeout)
        
        # 웹소켓에서 이미지를 받지 못했으면 히스토리 API 사용
        if not images:
            logger.info(f"히스토리 API로 이미지 조회 중... (프롬프트 ID: {prompt_id})")
            images = self._get_history_images(prompt_id, max_retries=10, retry_delay=3)
        
        # 타임아웃 체크
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(f"ComfyUI 이미지 생성 타임아웃 ({timeout}초)")
        
        if not images:
            raise ValueError(f"이미지가 생성되지 않았습니다. 프롬프트 ID: {prompt_id}")
        
        logger.info(f"이미지 생성 완료: {len(images)}개 파일")
        return images
    
    def _wait_via_websocket(self, prompt_id: str, timeout: int) -> List[str]:
        """
        웹소켓을 통해 이미지 생성 완료를 대기합니다.
        
        Args:
            prompt_id: 프롬프트 ID
            timeout: 타임아웃 (초)
            
        Returns:
            생성된 이미지 파일명 리스트 (실패 시 빈 리스트)
        """
        ws_url = f"{self.ws_url}?clientId={self.client_id}"
        
        images = []
        completed = False
        execution_error = None
        
        def on_message(ws, message):
            nonlocal images, completed, execution_error
            try:
                if isinstance(message, str):
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    logger.debug(f"웹소켓 메시지 수신: type={message_type}")
                    
                    if message_type == "executed":
                        exec_data = data.get("data", {})
                        exec_prompt_id = exec_data.get("prompt_id")
                        
                        if exec_prompt_id == prompt_id:
                            if "error" in exec_data:
                                execution_error = exec_data.get("error")
                                logger.error(f"ComfyUI 실행 에러: {execution_error}")
                                completed = True
                                ws.close()
                                return
                            
                            output = exec_data.get("output", {})
                            for node_id, node_output in output.items():
                                if "images" in node_output:
                                    for img_info in node_output["images"]:
                                        filename = img_info.get("filename")
                                        if filename:
                                            images.append(filename)
                            
                            completed = True
                            ws.close()
                            
                    elif message_type == "progress":
                        progress_data = data.get("data", {})
                        progress = progress_data.get("value", 0)
                        max_progress = progress_data.get("max", 0)
                        if max_progress > 0:
                            logger.debug(f"ComfyUI 진행률: {progress}/{max_progress}")
                        
            except Exception as e:
                logger.error(f"웹소켓 메시지 처리 오류: {e}")
        
        def on_error(ws, error):
            logger.debug(f"ComfyUI 웹소켓 오류: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.debug(f"ComfyUI 웹소켓 연결 종료")
        
        def on_open(ws):
            logger.debug("ComfyUI 웹소켓 연결 성공")
        
        try:
            logger.debug(f"웹소켓 연결 시도: {ws_url}")
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
            
        except Exception as e:
            logger.debug(f"웹소켓 연결 실패: {e}")
        
        if execution_error:
            raise RuntimeError(f"ComfyUI 실행 에러: {execution_error}")
        
        return images
    
    def download_result(self, filename: str, output_path: str) -> None:
        """
        ComfyUI에서 생성된 이미지를 다운로드합니다.
        
        Args:
            filename: 이미지 파일명
            output_path: 저장할 경로
            
        Raises:
            requests.RequestException: 다운로드 실패 시
        """
        image_url = f"{self.base_url}/view?filename={filename}&subfolder=&type=output"
        
        try:
            logger.debug(f"이미지 다운로드 시작: {filename}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"다운로드된 파일이 없습니다: {output_path}")
            
            logger.info(f"이미지 다운로드 완료: {output_path}")
            
        except requests.RequestException as e:
            logger.error(f"이미지 다운로드 실패 ({filename}): {e}")
            raise


def generate_images(scenes: List[Dict], output_dir: Path, retry_count: int = None) -> List[Dict]:
    """
    모든 장면의 이미지를 생성합니다.
    
    Args:
        scenes: 장면 리스트
        output_dir: 출력 디렉토리
        retry_count: 재시도 횟수
        
    Returns:
        이미지 경로가 업데이트된 장면 리스트
    """
    # 모의 모드 체크
    if Config.COMFYUI_USE_MOCK:
        from src.tools.comfyui_mock import generate_images_mock
        return generate_images_mock(scenes, output_dir)
    
    logger.info(f"[이미지 생성] {len(scenes)}개 장면 처리 시작")
    
    client = ComfyUIClient()
    workflow = client.load_workflow()
    
    updated_scenes = []
    retry_count = retry_count or Config.COMFYUI_RETRY_COUNT
    
    for idx, scene in enumerate(scenes):
        image_prompt = scene.get('image_prompt', '').strip()
        if not image_prompt:
            raise ValueError(f"장면 {idx+1}에 이미지 프롬프트가 없습니다.")
        
        logger.info(f"[이미지 생성] 장면 {idx+1} 처리 중: {image_prompt[:50]}...")
        
        # 워크플로우 프롬프트 업데이트
        updated_workflow = client.update_prompt(workflow, image_prompt)
        
        # 이미지 생성 (재시도 포함)
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                # 프롬프트 실행
                prompt_id = client.execute_prompt(updated_workflow, retry_count=0)  # execute_prompt 내부에서 재시도
                
                # 완료 대기
                image_filenames = client.wait_for_completion(prompt_id)
                
                if not image_filenames:
                    raise ValueError(f"장면 {idx+1}의 이미지가 생성되지 않았습니다.")
                
                # 첫 번째 이미지 다운로드
                image_filename = image_filenames[0]
                image_path = output_dir / f"image_scene_{idx+1}.png"
                client.download_result(image_filename, str(image_path))
                
                scene['image_path'] = str(image_path)
                updated_scenes.append(scene)
                logger.info(f"[이미지 생성] 장면 {idx+1} 완료: {image_path}")
                break
                
            except Exception as e:
                last_error = e
                if attempt < retry_count:
                    wait_time = (attempt + 1) * 3
                    logger.warning(f"이미지 생성 실패 (장면 {idx+1}, 시도 {attempt + 1}/{retry_count + 1}), {wait_time}초 후 재시도: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"이미지 생성 최종 실패 (장면 {idx+1}): {e}")
                    raise
    
    logger.info(f"[이미지 생성] 모든 이미지 생성 완료")
    return updated_scenes


def node_visual_generator(state: Dict) -> Dict:
    """
    LangGraph 노드 함수: 이미지 생성
    
    Args:
        state: 현재 상태 딕셔너리
        
    Returns:
        업데이트된 상태 딕셔너리
    """
    try:
        scenes = state.get("scenes", [])
        if not scenes:
            raise ValueError("상태에 'scenes'가 없거나 비어있습니다.")
        
        output_dir = Config.get_output_dir()
        updated_scenes = generate_images(scenes, output_dir)
        
        return {
            **state,
            "scenes": updated_scenes
        }
        
    except Exception as e:
        logger.error(f"[이미지 생성 노드] 오류: {e}", exc_info=True)
        raise

