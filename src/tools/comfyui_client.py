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
        
    def load_workflow(self, workflow_path: str = None) -> Dict:
        """
        ComfyUI 워크플로우 JSON 파일을 로드합니다.
        
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
            
            return default_workflow
        
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            logger.debug(f"워크플로우 로드 완료: {workflow_path}")
            return workflow
        except Exception as e:
            logger.error(f"워크플로우 로드 실패: {e}")
            raise
    
    def _create_default_workflow(self) -> Dict:
        """기본 워크플로우 생성"""
        return {
            "1": {
                "inputs": {
                    "text": "default prompt",
                    "clip": ["4", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Prompt)"}
            },
            "2": {
                "inputs": {
                    "text": "",
                    "clip": ["4", 0]
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
                    "ckpt_name": "v1-5-pruned-emaonly.safetensors"
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
                    "vae": ["4", 1]
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
        
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                logger.debug(f"ComfyUI 프롬프트 실행 요청 (시도 {attempt + 1}/{retry_count + 1})")
                response = requests.post(url, json=prompt_payload, timeout=10)
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
    
    def wait_for_completion(self, prompt_id: str, timeout: int = None) -> List[str]:
        """
        ComfyUI 웹소켓을 통해 이미지 생성 완료를 대기합니다.
        연결 실패 시 자동으로 재시도합니다.
        
        Args:
            prompt_id: 프롬프트 ID
            timeout: 타임아웃 (초, 기본값: self.timeout)
            
        Returns:
            생성된 이미지 파일명 리스트
            
        Raises:
            TimeoutError: 타임아웃 발생 시
            ConnectionError: 웹소켓 연결 실패 시
        """
        timeout = timeout or self.timeout
        ws_url = f"{self.ws_url}?clientId={self.client_id}"
        
        images = []
        start_time = time.time()
        completed = False
        
        def on_message(ws, message):
            nonlocal images, completed
            try:
                if isinstance(message, str):
                    data = json.loads(message)
                    
                    if data.get("type") == "executed":
                        if data.get("data", {}).get("prompt_id") == prompt_id:
                            # 실행 완료
                            output = data.get("data", {}).get("output", {})
                            for node_id, node_output in output.items():
                                if "images" in node_output:
                                    for img_info in node_output["images"]:
                                        filename = img_info.get("filename")
                                        if filename:
                                            images.append(filename)
                            completed = True
                            ws.close()
                            
                    elif data.get("type") == "progress":
                        # 진행 상황 출력
                        progress = data.get("data", {}).get("value", 0)
                        logger.debug(f"ComfyUI 진행률: {progress}%")
                        
            except Exception as e:
                logger.error(f"웹소켓 메시지 처리 오류: {e}")
        
        def on_error(ws, error):
            logger.error(f"ComfyUI 웹소켓 오류: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.debug("ComfyUI 웹소켓 연결 종료")
        
        # 웹소켓 연결 시도 (재시도 포함)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"웹소켓 연결 시도 {attempt + 1}/{max_retries}: {ws_url}")
                ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close
                )
                
                # 비동기 실행을 위한 래퍼
                ws.run_forever()
                
                if completed:
                    break
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"웹소켓 연결 실패 (시도 {attempt + 1}/{max_retries}), {wait_time}초 후 재시도: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"웹소켓 연결 최종 실패: {e}")
                    raise ConnectionError(f"웹소켓 연결 실패: {e}")
        
        # 타임아웃 체크
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(f"ComfyUI 이미지 생성 타임아웃 ({timeout}초)")
        
        if not images:
            raise ValueError(f"이미지가 생성되지 않았습니다. 프롬프트 ID: {prompt_id}")
        
        logger.info(f"이미지 생성 완료: {len(images)}개 파일")
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

