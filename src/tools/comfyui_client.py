"""
ComfyUI HTTP API 클라이언트 모듈

이 모듈은 ComfyUI 서버를 직접 실행하지 않고, HTTP API를 통해서만 통신합니다.
ComfyUI 서버는 사용자가 별도로 실행한 상태여야 하며,
COMFYUI_URL 환경변수에 지정된 HTTP 엔드포인트로만 요청을 보냅니다.

중요:
- ComfyUI 서버 프로세스는 이 코드에서 관리하지 않습니다
- 서버가 실행되지 않으면 명확한 에러 메시지를 표시합니다
- stderr/tqdm 충돌 문제를 피하기 위해 subprocess를 사용하지 않습니다
- tqdm 비활성화는 src/config.py에서 환경변수로 처리됩니다
"""

import copy
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import requests
import websocket

from src.config import Config

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """
    ComfyUI HTTP API 클라이언트
    
    이 클래스는 ComfyUI 서버를 직접 실행하지 않습니다.
    ComfyUI 서버는 사용자가 별도로 실행한 상태여야 하며,
    COMFYUI_URL 환경변수에 지정된 HTTP 엔드포인트로만 요청을 보냅니다.
    
    중요: ComfyUI 서버 프로세스는 이 코드에서 관리하지 않습니다.
    서버가 실행 중이지 않으면 명확한 에러 메시지를 표시합니다.
    """
    
    def __init__(self, base_url: str = None, ws_url: str = None, timeout: int = None):
        """
        ComfyUI 클라이언트 초기화
        
        서버는 외부에서 이미 실행 중이라고 가정합니다.
        서버를 직접 실행하거나 종료하지 않습니다.
        
        Args:
            base_url: ComfyUI 서버 URL (기본값: COMFYUI_URL 환경변수 또는 http://127.0.0.1:8188)
            ws_url: ComfyUI 웹소켓 URL (기본값: COMFYUI_WS_URL 환경변수 또는 ws://127.0.0.1:8188/ws)
            timeout: 타임아웃 (초, 기본값: COMFYUI_TIMEOUT 환경변수 또는 300)
        """
        # COMFYUI_URL 환경변수를 최우선으로 사용
        if base_url is None:
            base_url = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
        if ws_url is None:
            ws_url = os.getenv("COMFYUI_WS_URL", "ws://127.0.0.1:8188/ws")
        if timeout is None:
            timeout = Config.COMFYUI_TIMEOUT
        
        # URL 정규화 (끝의 슬래시 제거)
        self.base_url = base_url.rstrip("/")
        self.ws_url = ws_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())
        self.api_key = Config.COMFYUI_API_KEY if hasattr(Config, 'COMFYUI_API_KEY') else ""
        
        logger.debug(f"ComfyUI 클라이언트 초기화: {self.base_url} (서버는 외부에서 실행되어야 함)")
        
    def ping(self) -> bool:
        """
        ComfyUI 서버가 실행 중인지 확인합니다.
        
        서버는 외부에서 이미 실행되어야 합니다.
        이 메서드는 서버를 실행하지 않고, 단순히 헬스체크만 수행합니다.
        
        Returns:
            서버가 응답하면 True, 아니면 False
        """
        try:
            # system_stats 엔드포인트로 헬스체크
            url = f"{self.base_url}/system_stats"
            response = requests.get(url, timeout=5)
            return response.ok
        except Exception as e:
            logger.debug(f"ComfyUI 서버 헬스체크 실패: {e}")
            return False
    
    def check_health(self) -> None:
        """
        ComfyUI 서버 헬스체크를 수행하고, 실패 시 명확한 에러를 발생시킵니다.
        
        Raises:
            RuntimeError: 서버에 접속할 수 없는 경우
        """
        if not self.ping():
            raise RuntimeError(
                f"ComfyUI 서버({self.base_url})에 접속할 수 없습니다.\n"
                f"다음을 확인해주세요:\n"
                f"  1. ComfyUI 서버를 수동으로 실행했는지 확인\n"
                f"  2. .env 파일의 COMFYUI_URL 설정이 올바른지 확인 (현재: {self.base_url})\n"
                f"  3. ComfyUI 서버가 {self.base_url}에서 실행 중인지 확인"
            )
    
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
            
            # 연결 실패 시 명확한 에러 메시지
            if response.status_code == 404:
                raise RuntimeError(
                    f"ComfyUI 서버({self.base_url})를 찾을 수 없습니다. "
                    "서버가 실행 중인지 확인해주세요."
                )
            
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
                
        except requests.ConnectionError as e:
            error_msg = (
                f"ComfyUI 서버({self.base_url})에 연결할 수 없습니다.\n"
                f"서버를 수동으로 실행한 뒤 다시 시도해주세요.\n"
                f"연결 오류: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except requests.RequestException as e:
            logger.error(f"모델 목록 조회 실패: {e}")
            raise RuntimeError(
                f"ComfyUI API 호출 실패 ({self.base_url}): {e}\n"
                "서버가 실행 중이고 URL이 올바른지 확인해주세요."
            ) from e
    
    def get_available_node_types(self) -> List[str]:
        """
        ComfyUI에서 사용 가능한 노드 타입 목록을 가져옵니다.
        
        Returns:
            사용 가능한 노드 타입 리스트
        """
        try:
            url = f"{self.base_url}/object_info"
            response = requests.get(url, timeout=10)
            
            if not response.ok:
                if response.status_code == 404:
                    raise RuntimeError(
                        f"ComfyUI 서버({self.base_url})를 찾을 수 없습니다. "
                        "서버가 실행 중인지 확인해주세요."
                    )
                response.raise_for_status()
            
            object_info = response.json()
            node_types = list(object_info.keys())
            logger.debug(f"사용 가능한 노드 타입 {len(node_types)}개 발견")
            return node_types
                
        except requests.ConnectionError as e:
            error_msg = (
                f"ComfyUI 서버({self.base_url})에 연결할 수 없습니다.\n"
                f"서버를 수동으로 실행한 뒤 다시 시도해주세요."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except requests.RequestException as e:
            logger.error(f"노드 타입 목록 조회 실패: {e}")
            raise RuntimeError(
                f"ComfyUI API 호출 실패 ({self.base_url}): {e}\n"
                "서버가 실행 중이고 URL이 올바른지 확인해주세요."
            ) from e
    
    def check_node_exists(self, node_type: str) -> bool:
        """
        특정 노드 타입이 ComfyUI에 존재하는지 확인합니다.
        
        Args:
            node_type: 확인할 노드 타입 이름
            
        Returns:
            노드가 존재하면 True, 아니면 False
        """
        available_nodes = self.get_available_node_types()
        exists = node_type in available_nodes
        if not exists:
            # I2V 관련 노드 찾기
            i2v_nodes = [n for n in available_nodes if 'i2v' in n.lower() or 'imagetovideo' in n.lower() or 'video' in n.lower()]
            if i2v_nodes:
                logger.warning(
                    f"노드 '{node_type}'를 찾을 수 없습니다. "
                    f"사용 가능한 I2V 관련 노드: {', '.join(i2v_nodes[:5])}"
                )
            else:
                logger.warning(f"노드 '{node_type}'를 찾을 수 없습니다.")
        return exists
    
    def get_available_samplers(self) -> List[str]:
        """
        ComfyUI에서 사용 가능한 sampler 목록을 가져옵니다.
        ComfyUI Registry API (/object_info)를 사용합니다.
        
        Returns:
            사용 가능한 sampler 이름 리스트
        """
        try:
            url = f"{self.base_url}/object_info"
            response = requests.get(url, timeout=10)
            
            if not response.ok:
                if response.status_code == 404:
                    raise RuntimeError(
                        f"ComfyUI 서버({self.base_url})를 찾을 수 없습니다. "
                        "서버가 실행 중인지 확인해주세요."
                    )
                response.raise_for_status()
            
            object_info = response.json()
            ksampler_info = object_info.get("KSampler", {})
            inputs = ksampler_info.get("input", {})
            required = inputs.get("required", {})
            sampler_name_info = required.get("sampler_name", [])
            
            # sampler_name_info는 보통 [["sampler1", "sampler2", ...], {"tooltip": "..."}] 형태
            if isinstance(sampler_name_info, list) and len(sampler_name_info) > 0:
                samplers = sampler_name_info[0] if isinstance(sampler_name_info[0], list) else sampler_name_info
                logger.debug(f"사용 가능한 sampler {len(samplers)}개 발견")
                return samplers
            
            return []
                
        except requests.ConnectionError as e:
            error_msg = (
                f"ComfyUI 서버({self.base_url})에 연결할 수 없습니다.\n"
                f"서버를 수동으로 실행한 뒤 다시 시도해주세요."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except requests.RequestException as e:
            logger.error(f"Sampler 목록 조회 실패: {e}")
            raise RuntimeError(
                f"ComfyUI API 호출 실패 ({self.base_url}): {e}\n"
                "서버가 실행 중이고 URL이 올바른지 확인해주세요."
            ) from e
    
    def get_recommended_samplers(self) -> List[str]:
        """
        tqdm 문제가 없는 안정적인 sampler 목록을 반환합니다.
        res_multistep 계열은 제외합니다.
        
        Returns:
            권장 sampler 이름 리스트
        """
        all_samplers = self.get_available_samplers()
        # res_multistep 계열 제외 (tqdm 문제 발생 가능)
        recommended = [
            s for s in all_samplers 
            if not s.startswith("res_multistep")
        ]
        
        # 우선순위: 고품질 sampler 우선
        priority_order = [
            "dpmpp_2m",
            "dpmpp_2m_sde",
            "dpmpp_2m_sde_gpu",
            "euler",
            "euler_ancestral",
            "dpmpp_sde",
            "dpmpp_sde_gpu",
            "lcm",
            "heun",
            "dpm_2",
            "dpm_2_ancestral",
        ]
        
        # 우선순위대로 정렬
        recommended_sorted = []
        for priority in priority_order:
            if priority in recommended:
                recommended_sorted.append(priority)
                recommended.remove(priority)
        
        # 나머지 추가
        recommended_sorted.extend(sorted(recommended))
        
        return recommended_sorted
    
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
        
        # KSampler 노드 찾아서 res_multistep 계열 sampler 자동 변경
        for node_id, node_data in workflow.items():
            class_type = node_data.get("class_type")
            if class_type == "KSampler":
                inputs = node_data.get("inputs", {})
                sampler_name = inputs.get("sampler_name")
                # res_multistep 계열 sampler는 tqdm 문제가 있으므로 자동 변경
                if sampler_name and isinstance(sampler_name, str) and "res_multistep" in sampler_name.lower():
                    try:
                        recommended_samplers = self.get_recommended_samplers()
                        if recommended_samplers:
                            # 첫 번째 권장 sampler로 변경
                            new_sampler = recommended_samplers[0]
                            inputs["sampler_name"] = new_sampler
                            logger.warning(
                                f"워크플로우 노드 {node_id}: sampler '{sampler_name}'를 '{new_sampler}'로 자동 변경했습니다. "
                                f"(res_multistep 계열은 tqdm 문제가 발생할 수 있습니다)"
                            )
                        else:
                            # 권장 sampler를 가져올 수 없으면 기본값 사용
                            inputs["sampler_name"] = "euler"
                            logger.warning(
                                f"워크플로우 노드 {node_id}: sampler '{sampler_name}'를 'euler'로 자동 변경했습니다."
                            )
                    except Exception as e:
                        logger.warning(
                            f"워크플로우 노드 {node_id}: sampler 자동 변경 실패 ({e}), "
                            f"기본값 'euler'로 변경합니다."
                        )
                        inputs["sampler_name"] = "euler"
        
        # CheckpointLoaderSimple 또는 UNETLoader 노드 찾아서 모델 자동 설정
        checkpoint_node_id = None
        unet_node_id = None
        
        for node_id, node_data in workflow.items():
            class_type = node_data.get("class_type")
            if class_type == "CheckpointLoaderSimple":
                checkpoint_node_id = node_id
            elif class_type == "UNETLoader":
                unet_node_id = node_id
        
        # CheckpointLoaderSimple이 있으면 기존 방식 사용
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
        
        # UNETLoader가 있으면 Wan2.2 방식 사용
        if unet_node_id:
            # 환경 변수에서 지정된 UNet 모델이 있으면 사용 (I2V_UNET_NAME)
            if hasattr(Config, 'I2V_UNET_NAME') and Config.I2V_UNET_NAME:
                workflow[unet_node_id]["inputs"]["unet_name"] = Config.I2V_UNET_NAME
                logger.info(f"환경 변수에서 지정된 UNet 모델 사용: {Config.I2V_UNET_NAME}")
            else:
                # 기본값 유지
                current_unet = workflow[unet_node_id]["inputs"].get("unet_name", "")
                if current_unet:
                    logger.info(f"UNet 모델 사용: {current_unet}")
                else:
                    logger.warning("UNet 모델이 설정되지 않았습니다.")
        
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
    
    def update_prompt(self, workflow: Dict, prompt: str, ensure_style_consistency: bool = True) -> Dict:
        """
        워크플로우에서 프롬프트 노드를 찾아 프롬프트를 업데이트합니다.
        딥카피를 사용하여 원본 워크플로우를 보호합니다.
        
        Args:
            workflow: ComfyUI 워크플로우 딕셔너리
            prompt: 새로운 프롬프트
            ensure_style_consistency: 스타일 일관성 태그 자동 추가 여부 (기본값: True)
            
        Returns:
            업데이트된 워크플로우 (딥카피)
        """
        if not prompt or not prompt.strip():
            raise ValueError("프롬프트가 비어있습니다.")
        
        # 딥카피 생성
        updated_workflow = copy.deepcopy(workflow)
        
        # 고품질 스타일 태그 추가 (이미 포함되어 있지 않은 경우)
        enhanced_prompt = prompt
        if ensure_style_consistency:
            # 고품질 태그 우선 확인
            quality_tags = ["masterpiece", "best quality", "ultra detailed"]
            style_tags = ["anime style", "consistent art style", "JANKU style", "detailed anime illustration"]
            
            prompt_lower = prompt.lower()
            
            # 고품질 태그가 없으면 추가
            missing_quality_tags = [tag for tag in quality_tags if tag.lower() not in prompt_lower]
            missing_style_tags = [tag for tag in style_tags if tag.lower() not in prompt_lower]
            
            if missing_quality_tags or missing_style_tags:
                all_missing_tags = missing_quality_tags + missing_style_tags
                enhanced_prompt = ", ".join(all_missing_tags) + ", " + prompt
                logger.debug(f"고품질 스타일 태그 추가: {', '.join(all_missing_tags)}")
        
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
        
        updated_workflow[prompt_node_id]["inputs"]["text"] = enhanced_prompt
        logger.debug(f"프롬프트 업데이트: 노드 {prompt_node_id} - {enhanced_prompt[:80]}...")
        
        return updated_workflow
    
    def _save_prompt_to_file(self, prompt_id: str, workflow: Dict) -> Path:
        """
        프롬프트 워크플로우를 파일로 저장합니다 (디버깅용).
        
        Args:
            prompt_id: 프롬프트 ID
            workflow: 워크플로우 딕셔너리
            
        Returns:
            저장된 파일 경로
        """
        logs_dir = Path("logs") / "comfyui_prompts"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        prompt_file = logs_dir / f"{prompt_id}.json"
        try:
            with open(prompt_file, 'w', encoding='utf-8') as f:
                json.dump(workflow, f, indent=2, ensure_ascii=False)
            logger.debug(f"프롬프트 워크플로우 저장: {prompt_file}")
            return prompt_file
        except Exception as e:
            logger.warning(f"프롬프트 워크플로우 저장 실패: {e}")
            return prompt_file
    
    def _save_history_to_file(self, prompt_id: str, history_data: Dict) -> Path:
        """
        ComfyUI history 응답을 파일로 저장합니다 (디버깅용).
        
        Args:
            prompt_id: 프롬프트 ID
            history_data: history 딕셔너리 (전체 또는 prompt_id에 해당하는 부분)
            
        Returns:
            저장된 파일 경로
        """
        logs_dir = Path("logs") / "comfyui_history"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        history_file = logs_dir / f"{prompt_id}.json"
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"ComfyUI history 저장: {history_file}")
            return history_file
        except Exception as e:
            logger.warning(f"ComfyUI history 저장 실패: {e}")
            return history_file
    
    def _extract_error_details(self, error_data: Dict, prompt_data: Dict = None) -> Dict:
        """
        ComfyUI 에러 데이터에서 상세 정보를 추출합니다.
        
        Args:
            error_data: execution_error 메시지의 데이터 부분
            prompt_data: 프롬프트 전체 데이터 (inputs 추출용)
            
        Returns:
            에러 상세 정보 딕셔너리
        """
        details = {
            "node_id": error_data.get("node_id", ""),
            "node_type": error_data.get("node_type", ""),
            "exception_type": error_data.get("exception_type", ""),
            "exception_message": error_data.get("exception_message", ""),
            "traceback": error_data.get("traceback", []),
            "current_inputs": error_data.get("current_inputs", {}),
            "current_outputs": error_data.get("current_outputs", []),
        }
        
        # KSampler인 경우 주요 입력값 추출
        if details["node_type"] == "KSampler" and prompt_data:
            prompt = prompt_data.get("prompt", {})
            node_id = details["node_id"]
            
            # node_id를 문자열로 변환 (prompt의 키는 문자열)
            node_id_str = str(node_id) if node_id is not None else ""
            
            # prompt에서 노드 찾기 (문자열 키로 시도)
            node_config = None
            if node_id_str in prompt:
                node_config = prompt[node_id_str]
            elif node_id in prompt:
                node_config = prompt[node_id]
            
            if node_config and isinstance(node_config, dict):
                inputs = node_config.get("inputs", {})
                # 리스트인 경우 첫 번째 값 추출, 아니면 그대로 사용
                def extract_value(v):
                    if isinstance(v, list) and len(v) > 0:
                        return v[0]
                    return v
                
                details["ksampler_inputs"] = {
                    "seed": extract_value(inputs.get("seed")),
                    "steps": extract_value(inputs.get("steps")),
                    "cfg": extract_value(inputs.get("cfg")),
                    "sampler_name": extract_value(inputs.get("sampler_name")),
                    "scheduler": extract_value(inputs.get("scheduler")),
                    "denoise": extract_value(inputs.get("denoise")),
                }
            else:
                # prompt에서 찾지 못한 경우, current_inputs에서 추출 시도
                current_inputs = details.get("current_inputs", {})
                if current_inputs:
                    # 리스트인 경우 첫 번째 값 추출, 아니면 그대로 사용
                    def extract_value(v):
                        if isinstance(v, list) and len(v) > 0:
                            return v[0]
                        return v
                    
                    details["ksampler_inputs"] = {
                        "seed": extract_value(current_inputs.get("seed")),
                        "steps": extract_value(current_inputs.get("steps")),
                        "cfg": extract_value(current_inputs.get("cfg")),
                        "sampler_name": extract_value(current_inputs.get("sampler_name")),
                        "scheduler": extract_value(current_inputs.get("scheduler")),
                        "denoise": extract_value(current_inputs.get("denoise")),
                    }
        
        return details
    
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
                
                # 연결 실패 시 명확한 에러 메시지
                if isinstance(response, requests.ConnectionError) or (hasattr(response, 'status_code') and response.status_code == 0):
                    raise RuntimeError(
                        f"ComfyUI 서버({self.base_url})에 연결할 수 없습니다.\n"
                        f"서버를 수동으로 실행한 뒤 다시 시도해주세요."
                    )
                
                # 400 오류인 경우 상세 정보 로깅
                if response.status_code == 400:
                    try:
                        error_detail = response.json()
                        logger.error(f"ComfyUI 400 오류 상세: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                    except:
                        logger.error(f"ComfyUI 400 오류 응답 본문: {response.text[:500]}")
                
                # 404 오류인 경우 서버 미실행으로 간주
                if response.status_code == 404:
                    raise RuntimeError(
                        f"ComfyUI 서버({self.base_url})를 찾을 수 없습니다.\n"
                        f"서버가 실행 중인지 확인해주세요."
                    )
                
                response.raise_for_status()
                
                result = response.json()
                prompt_id = result.get("prompt_id")
                
                if not prompt_id:
                    raise ValueError("ComfyUI에서 prompt_id를 받지 못했습니다.")
                
                # 프롬프트 워크플로우를 파일로 저장 (디버깅용)
                prompt_file = self._save_prompt_to_file(prompt_id, workflow)
                logger.debug(f"프롬프트 워크플로우 저장됨: {prompt_file}")
                
                # DEBUG: 워크플로우 전체 구조 로깅
                workflow_debug = f"[워크플로우 전체] 프롬프트 ID {prompt_id} 실행 워크플로우:\n{json.dumps(workflow, indent=2, ensure_ascii=False)}"
                logger.debug("=" * 80)
                logger.debug(workflow_debug)
                logger.debug("=" * 80)
                
                # 파일에 직접 기록 (로거 설정과 무관하게)
                try:
                    log_dir = Path("logs")
                    log_dir.mkdir(exist_ok=True)
                    debug_log_file = log_dir / "comfyui_client_debug.log"
                    with open(debug_log_file, "a", encoding="utf-8") as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DEBUG - 워크플로우 전체\n")
                        f.write(workflow_debug)
                        f.write(f"\n{'='*80}\n")
                except Exception:
                    pass
                
                logger.info(f"프롬프트 ID: {prompt_id}")
                return prompt_id
                
            except requests.ConnectionError as e:
                last_error = RuntimeError(
                    f"ComfyUI 서버({self.base_url})에 연결할 수 없습니다.\n"
                    f"서버를 수동으로 실행한 뒤 다시 시도해주세요.\n"
                    f"연결 오류: {e}"
                )
                if attempt < retry_count:
                    wait_time = (attempt + 1) * 2  # 지수 백오프
                    logger.warning(f"ComfyUI 서버 연결 실패 (시도 {attempt + 1}/{retry_count + 1}), {wait_time}초 후 재시도")
                    time.sleep(wait_time)
                else:
                    logger.error(f"ComfyUI 서버 연결 최종 실패")
            except requests.RequestException as e:
                last_error = e
                if attempt < retry_count:
                    wait_time = (attempt + 1) * 2  # 지수 백오프
                    logger.warning(f"ComfyUI API 호출 실패 (시도 {attempt + 1}/{retry_count + 1}), {wait_time}초 후 재시도: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"ComfyUI API 호출 최종 실패: {e}")
        
        if last_error:
            raise last_error
        else:
            raise RuntimeError("ComfyUI 프롬프트 실행 실패: 알 수 없는 오류")
    
    def _is_fatal_error(self, error_data: Dict) -> bool:
        """
        에러가 치명적인지(재시도 불가능한지) 판단합니다.
        
        Args:
            error_data: execution_error 메시지의 데이터 부분
            
        Returns:
            치명적 에러면 True, 아니면 False
        """
        exception_type = error_data.get("exception_type", "")
        exception_message = str(error_data.get("exception_message", ""))
        
        # OSError [Errno 22] Invalid argument는 치명적 에러
        if exception_type == "OSError" and "[Errno 22] Invalid argument" in exception_message:
            return True
        
        # 다른 치명적 에러 타입도 여기에 추가 가능
        # 예: ValueError, TypeError 등 입력값 문제
        
        return False
    
    def _get_history_images(self, prompt_id: str, max_retries: int = 10, retry_delay: int = 3) -> List[str]:
        """
        ComfyUI API를 통해 히스토리에서 이미지 파일명을 조회합니다.
        이미지 생성 완료를 기다리기 위해 여러 번 재시도합니다.
        
        치명적 에러(OSError Errno 22 등)가 발생하면 재시도하지 않고 즉시 예외를 발생시킵니다.
        
        Args:
            prompt_id: 프롬프트 ID
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간 대기 시간 (초)
            
        Returns:
            생성된 이미지 파일명 리스트
            
        Raises:
            RuntimeError: 치명적 에러 발생 시 또는 최대 재시도 도달 시
        """
        logger.info(f"히스토리 API로 이미지 조회 시작 (프롬프트 ID: {prompt_id}, 최대 {max_retries}회 시도)")
        
        for attempt in range(max_retries):
            try:
                # ComfyUI 히스토리 API는 전체 히스토리를 반환
                url = f"{self.base_url}/history"
                logger.debug(f"히스토리 API 호출: {url} (시도 {attempt + 1}/{max_retries})")
                response = requests.get(url, timeout=10)
                
                # 연결 실패 시 명확한 에러 메시지
                if response.status_code == 404:
                    raise RuntimeError(
                        f"ComfyUI 서버({self.base_url})를 찾을 수 없습니다. "
                        "서버가 실행 중인지 확인해주세요."
                    )
                
                response.raise_for_status()
                
                history = response.json()
                
                # 히스토리 전체를 파일로 저장 (디버깅용, 첫 번째 시도에서만)
                if attempt == 0 and prompt_id in history:
                    history_file = self._save_history_to_file(prompt_id, history)
                    logger.debug(f"ComfyUI history 저장됨: {history_file}")
                    
                    # DEBUG: 히스토리 전체 구조 로깅 (첫 번째 시도에서만)
                    logger.debug("=" * 80)
                    logger.debug(f"[ComfyUI History 전체] 프롬프트 ID {prompt_id} (첫 번째 시도):")
                    logger.debug(json.dumps(history.get(prompt_id, {}), indent=2, ensure_ascii=False))
                    logger.debug("=" * 80)
                
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
                    status = prompt_data.get("status", {})
                    prompt_workflow = prompt_data.get("prompt", {})
                    
                    # 상태 확인 (실행 실패 여부)
                    error_detected = False
                    error_msg = None
                    error_details = None
                    is_fatal = False
                    
                    # status가 딕셔너리인 경우 messages 배열 확인
                    if isinstance(status, dict):
                        status_str = status.get("status_str", "")
                        completed = status.get("completed", True)
                        messages = status.get("messages", [])
                        
                        # messages 배열에서 execution_error 찾기
                        for msg in messages:
                            if isinstance(msg, list) and len(msg) > 0:
                                msg_type = msg[0]
                                if msg_type == "execution_error" and len(msg) > 1:
                                    error_data = msg[1] if isinstance(msg[1], dict) else {}
                                    
                                    # 에러 상세 정보 추출
                                    error_details = self._extract_error_details(error_data, prompt_data)
                                    
                                    # 치명적 에러인지 확인
                                    is_fatal = self._is_fatal_error(error_data)
                                    
                                    # DEBUG: 에러 노드 전체 데이터 출력 (강화)
                                    # 로거 레벨 확인 및 강제 출력
                                    debug_msg_parts = []
                                    debug_msg_parts.append("=" * 80)
                                    debug_msg_parts.append(f"[에러 상세] 프롬프트 ID: {prompt_id}")
                                    debug_msg_parts.append(f"[에러 상세] 노드 ID: {error_details['node_id']}")
                                    debug_msg_parts.append(f"[에러 상세] 노드 타입: {error_details['node_type']}")
                                    debug_msg_parts.append(f"[에러 상세] 예외 타입: {error_details['exception_type']}")
                                    debug_msg_parts.append(f"[에러 상세] 예외 메시지: {error_details['exception_message']}")
                                    
                                    # current_inputs 전체 출력
                                    if error_details.get('current_inputs'):
                                        debug_msg_parts.append(f"[에러 상세] current_inputs 전체:")
                                        debug_msg_parts.append(json.dumps(error_details['current_inputs'], indent=2, ensure_ascii=False))
                                    
                                    # current_outputs 출력
                                    if error_details.get('current_outputs'):
                                        debug_msg_parts.append(f"[에러 상세] current_outputs:")
                                        debug_msg_parts.append(json.dumps(error_details['current_outputs'], indent=2, ensure_ascii=False))
                                    
                                    # KSampler 입력값 상세 출력
                                    if error_details.get('ksampler_inputs'):
                                        debug_msg_parts.append(f"[에러 상세] KSampler 입력값 (추출됨):")
                                        debug_msg_parts.append(json.dumps(error_details['ksampler_inputs'], indent=2, ensure_ascii=False))
                                    
                                    # traceback 전체 출력
                                    if error_details.get('traceback'):
                                        debug_msg_parts.append(f"[에러 상세] traceback 전체 ({len(error_details['traceback'])}줄):")
                                        for i, line in enumerate(error_details['traceback'], 1):
                                            debug_msg_parts.append(f"  [{i:3d}] {line}")
                                    
                                    # 워크플로우에서 해당 노드의 전체 설정 출력
                                    if prompt_data and prompt_data.get("prompt"):
                                        prompt_workflow = prompt_data.get("prompt", {})
                                        node_id_str = str(error_details['node_id'])
                                        node_full_config = None
                                        
                                        if node_id_str in prompt_workflow:
                                            node_full_config = prompt_workflow[node_id_str]
                                            debug_msg_parts.append(f"[에러 상세] 워크플로우에서 노드 {node_id_str} 전체 설정:")
                                            debug_msg_parts.append(json.dumps(node_full_config, indent=2, ensure_ascii=False))
                                        
                                        # 연결된 노드들도 확인
                                        if node_full_config and isinstance(node_full_config, dict):
                                            debug_msg_parts.append(f"[에러 상세] 연결된 노드 확인:")
                                            inputs = node_full_config.get("inputs", {})
                                            for input_key, input_value in inputs.items():
                                                if isinstance(input_value, list) and len(input_value) >= 2:
                                                    connected_node_id = str(input_value[0])
                                                    output_index = input_value[1] if len(input_value) > 1 else 0
                                                    if connected_node_id in prompt_workflow:
                                                        connected_node = prompt_workflow[connected_node_id]
                                                        connected_type = connected_node.get("class_type", "unknown")
                                                        connected_inputs = connected_node.get("inputs", {})
                                                        debug_msg_parts.append(f"  - {input_key} -> 노드 {connected_node_id} ({connected_type}) 출력 {output_index}")
                                                        # 연결된 노드의 주요 입력값도 출력
                                                        if connected_type == "ModelSamplingAuraFlow" and "shift" in connected_inputs:
                                                            debug_msg_parts.append(f"      shift: {connected_inputs.get('shift')}")
                                                        elif connected_type == "CheckpointLoaderSimple" and "ckpt_name" in connected_inputs:
                                                            debug_msg_parts.append(f"      ckpt_name: {connected_inputs.get('ckpt_name')}")
                                                        elif connected_type == "EmptySD3LatentImage":
                                                            debug_msg_parts.append(f"      width: {connected_inputs.get('width')}, height: {connected_inputs.get('height')}")
                                                elif not isinstance(input_value, list):
                                                    # 리스트가 아닌 직접 값인 경우
                                                    debug_msg_parts.append(f"  - {input_key}: {input_value}")
                                    
                                    debug_msg_parts.append("=" * 80)
                                    
                                    # DEBUG 로그 출력 (로거 레벨과 관계없이 파일에 기록)
                                    debug_msg = "\n".join(debug_msg_parts)
                                    logger.debug(debug_msg)
                                    
                                    # 파일에 직접 기록 (로거 설정과 무관하게)
                                    try:
                                        log_dir = Path("logs")
                                        log_dir.mkdir(exist_ok=True)
                                        debug_log_file = log_dir / "comfyui_client_debug.log"
                                        with open(debug_log_file, "a", encoding="utf-8") as f:
                                            f.write(f"\n{'='*80}\n")
                                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DEBUG - 에러 상세 정보\n")
                                            f.write(debug_msg)
                                            f.write(f"\n{'='*80}\n")
                                    except Exception as e:
                                        # 파일 기록 실패해도 계속 진행
                                        pass
                                    
                                    # WARNING/ERROR: 핵심 정보만 출력
                                    error_msg = (
                                        f"ComfyUI 실행 실패\n"
                                        f"  프롬프트 ID: {prompt_id}\n"
                                        f"  노드 ID: {error_details['node_id']}\n"
                                        f"  노드 타입: {error_details['node_type']}\n"
                                        f"  예외 타입: {error_details['exception_type']}\n"
                                        f"  예외 메시지: {error_details['exception_message']}"
                                    )
                                    
                                    # KSampler인 경우 주요 입력값 추가
                                    if error_details.get('ksampler_inputs'):
                                        ksampler = error_details['ksampler_inputs']
                                        error_msg += (
                                            f"\n  KSampler 입력값:\n"
                                            f"    seed: {ksampler.get('seed')}\n"
                                            f"    steps: {ksampler.get('steps')}\n"
                                            f"    cfg: {ksampler.get('cfg')}\n"
                                            f"    sampler_name: {ksampler.get('sampler_name')}\n"
                                            f"    scheduler: {ksampler.get('scheduler')}\n"
                                            f"    denoise: {ksampler.get('denoise')}"
                                        )
                                    
                                    # 치명적 에러인 경우 추가 안내
                                    if is_fatal:
                                        # OSError [Errno 22]는 tqdm 문제일 가능성이 높음
                                        if "Errno 22" in error_details.get('exception_message', ''):
                                            error_msg += (
                                                f"\n\n[중요] 이 에러는 ComfyUI 서버의 tqdm/stderr 충돌 문제입니다.\n"
                                                f"해결 방법:\n"
                                                f"  1. ComfyUI 서버를 완전히 종료하세요\n"
                                                f"  2. PowerShell에서 다음 명령으로 환경변수를 설정하고 서버를 재시작하세요:\n"
                                                f"     $env:TQDM_DISABLE='1'\n"
                                                f"     $env:TQDM_MININTERVAL='999999'\n"
                                                f"     $env:TQDM_NCOLS='0'\n"
                                                f"     python main.py --port 8188\n"
                                                f"  3. 또는 ComfyUI 서버 시작 스크립트에 환경변수를 추가하세요\n"
                                                f"\n"
                                                f"이 에러가 발생한 프롬프트 전체 내용은 logs/comfyui_prompts/{prompt_id}.json에 저장되어 있습니다.\n"
                                                f"상세 DEBUG 로그는 logs/comfyui_client_debug.log를 확인하세요."
                                            )
                                        else:
                                            error_msg += (
                                                f"\n\n[중요] 이 에러는 치명적 에러로, 같은 프롬프트로 재시도해도 해결되지 않습니다.\n"
                                                f"ComfyUI 서버 측 문제일 수도 있지만, 현재 프롬프트나 노드 설정이 잘못되었을 가능성이 높습니다.\n"
                                                f"에러 로그에 출력된 노드 입력값(seed, steps, cfg 등)을 먼저 확인해보세요.\n"
                                                f"이 에러가 발생한 프롬프트 전체 내용은 logs/comfyui_prompts/{prompt_id}.json에 저장되어 있습니다.\n"
                                                f"상세 DEBUG 로그는 logs/comfyui_client_debug.log를 확인하세요."
                                            )
                                    else:
                                        error_msg += (
                                            f"\n\n[해결 방법] 이 에러는 ComfyUI 서버 측 문제일 수 있습니다. 다음을 시도해보세요:\n"
                                            f"  1. ComfyUI 서버를 재시작하세요\n"
                                            f"  2. ComfyUI-Manager 커스텀 노드를 업데이트하세요\n"
                                            f"  3. ComfyUI 서버 로그를 확인하세요"
                                        )
                                    
                                    error_detected = True
                                    break
                        
                        # messages에 없으면 status_str 확인
                        if not error_detected and (not completed or "error" in status_str.lower() or "failed" in status_str.lower()):
                            error_msg = f"ComfyUI 실행 실패: {status_str}"
                            error_detected = True
                    
                    # status가 리스트인 경우 (이전 형식)
                    elif isinstance(status, list) and len(status) > 0:
                        status_info = status[0] if isinstance(status[0], dict) else status
                        if isinstance(status_info, dict):
                            status_str = status_info.get("status_str", "")
                            completed = status_info.get("completed", True)
                            if not completed or "error" in status_str.lower() or "failed" in status_str.lower():
                                # 에러 메시지 추출
                                exception_message = status_info.get("exception_message", "")
                                exception_type = status_info.get("exception_type", "")
                                node_id = status_info.get("node_id", "")
                                node_type = status_info.get("node_type", "")
                                
                                error_data = {
                                    "node_id": node_id,
                                    "node_type": node_type,
                                    "exception_type": exception_type,
                                    "exception_message": exception_message,
                                }
                                
                                error_details = self._extract_error_details(error_data, prompt_data)
                                is_fatal = self._is_fatal_error(error_data)
                                
                                error_msg = (
                                    f"ComfyUI 실행 실패\n"
                                    f"  프롬프트 ID: {prompt_id}\n"
                                    f"  노드 ID: {node_id}\n"
                                    f"  노드 타입: {node_type}\n"
                                    f"  예외 타입: {exception_type}\n"
                                    f"  예외 메시지: {exception_message}"
                                )
                                
                                if is_fatal:
                                    error_msg += (
                                        f"\n\n[중요] 이 에러는 치명적 에러로, 같은 프롬프트로 재시도해도 해결되지 않습니다.\n"
                                        f"이 에러가 발생한 프롬프트 전체 내용은 logs/comfyui_prompts/{prompt_id}.json에 저장되어 있습니다."
                                    )
                                
                                error_detected = True
                    
                    # 에러 노드의 outputs 전체를 DEBUG 로그로 덤프 (강화)
                    if error_detected and outputs:
                        debug_outputs = []
                        debug_outputs.append("=" * 80)
                        debug_outputs.append(f"[에러 노드 출력] 프롬프트 ID {prompt_id}의 모든 outputs:")
                        for node_id, node_output in outputs.items():
                            if isinstance(node_output, dict):
                                debug_outputs.append(f"  노드 {node_id} outputs:")
                                debug_outputs.append(json.dumps(node_output, indent=2, ensure_ascii=False))
                            else:
                                debug_outputs.append(f"  노드 {node_id} outputs (타입: {type(node_output)}): {node_output}")
                        debug_outputs.append("=" * 80)
                        debug_msg = "\n".join(debug_outputs)
                        logger.debug(debug_msg)
                        
                        # 파일에 직접 기록
                        try:
                            log_dir = Path("logs")
                            log_dir.mkdir(exist_ok=True)
                            debug_log_file = log_dir / "comfyui_client_debug.log"
                            with open(debug_log_file, "a", encoding="utf-8") as f:
                                f.write(f"\n{debug_msg}\n")
                        except Exception:
                            pass
                    
                    # 히스토리 전체를 DEBUG 로그로 출력 (에러 발생 시, 강화)
                    if error_detected:
                        debug_history = []
                        debug_history.append("=" * 80)
                        debug_history.append(f"[ComfyUI History 전체] 프롬프트 ID {prompt_id}:")
                        debug_history.append(json.dumps(prompt_data, indent=2, ensure_ascii=False))
                        debug_history.append("=" * 80)
                        
                        # status 상세 정보도 별도로 출력
                        if status:
                            debug_history.append(f"[ComfyUI Status 상세] 프롬프트 ID {prompt_id}:")
                            debug_history.append(json.dumps(status, indent=2, ensure_ascii=False))
                            debug_history.append("=" * 80)
                        
                        debug_msg = "\n".join(debug_history)
                        logger.debug(debug_msg)
                        
                        # 파일에 직접 기록
                        try:
                            log_dir = Path("logs")
                            log_dir.mkdir(exist_ok=True)
                            debug_log_file = log_dir / "comfyui_client_debug.log"
                            with open(debug_log_file, "a", encoding="utf-8") as f:
                                f.write(f"\n{debug_msg}\n")
                        except Exception:
                            pass
                    
                    if error_detected and error_msg:
                        logger.error(f"프롬프트 ID {prompt_id} {error_msg}")
                        
                        # 치명적 에러는 즉시 예외 발생 (재시도하지 않음)
                        if is_fatal:
                            raise RuntimeError(error_msg)
                        
                        # 치명적이지 않은 에러는 재시도 가능
                        if attempt < max_retries - 1:
                            logger.warning(f"에러 발생, {retry_delay}초 후 재시도... (시도 {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                        else:
                            # 최대 재시도 도달 시 예외 발생
                            raise RuntimeError(error_msg)
                    
                    logger.info(f"히스토리에서 프롬프트 ID {prompt_id} 발견, 출력 노드 수: {len(outputs)}, 상태: {status}")
                    
                    # outputs 구조 상세 로깅
                    if outputs:
                        logger.debug(f"출력 노드 ID 목록: {list(outputs.keys())}")
                        for node_id, node_output in outputs.items():
                            logger.debug(f"노드 {node_id} 출력 구조: {list(node_output.keys()) if isinstance(node_output, dict) else type(node_output)}")
                            
                            # images 필드 확인
                            if isinstance(node_output, dict) and "images" in node_output:
                                images_list = node_output["images"]
                                logger.debug(f"노드 {node_id}의 이미지 리스트 타입: {type(images_list)}, 길이: {len(images_list) if isinstance(images_list, list) else 'N/A'}")
                                
                                if isinstance(images_list, list):
                                    for img_info in images_list:
                                        if isinstance(img_info, dict):
                                            filename = img_info.get("filename")
                                            if filename:
                                                images.append(filename)
                                                logger.info(f"히스토리에서 이미지 발견: {filename} (노드 {node_id})")
                                        elif isinstance(img_info, str):
                                            # 이미지 정보가 문자열로 직접 들어올 수도 있음
                                            images.append(img_info)
                                            logger.info(f"히스토리에서 이미지 발견: {img_info} (노드 {node_id})")
                            elif isinstance(node_output, dict):
                                # 다른 필드에서 이미지 찾기 시도
                                for key, value in node_output.items():
                                    if "image" in key.lower() and isinstance(value, list):
                                        for item in value:
                                            if isinstance(item, dict) and "filename" in item:
                                                filename = item.get("filename")
                                                if filename:
                                                    images.append(filename)
                                                    logger.info(f"히스토리에서 이미지 발견 (키: {key}): {filename} (노드 {node_id})")
                
                if images:
                    logger.info(f"히스토리 API에서 {len(images)}개 이미지 발견 (시도 {attempt + 1}/{max_retries})")
                    return images
                else:
                    if prompt_id in history:
                        prompt_data = history[prompt_id]
                        outputs = prompt_data.get("outputs", {})
                        status = prompt_data.get("status", {})
                        
                        # 상태에서 에러 확인 (이미 위에서 처리했지만, 재확인)
                        has_error = False
                        if isinstance(status, list) and len(status) > 0:
                            for status_item in status:
                                if isinstance(status_item, list) and len(status_item) > 1:
                                    status_type = status_item[0]
                                    status_data = status_item[1] if isinstance(status_item[1], dict) else {}
                                    if status_type == "execution_error":
                                        has_error = True
                                        break
                        
                        if has_error:
                            # 에러가 있으면 위에서 이미 예외가 발생했을 것이므로, 여기서는 로깅만
                            logger.warning(
                                f"프롬프트 ID {prompt_id}는 에러로 인해 이미지가 생성되지 않았습니다. "
                                f"출력 구조: {list(prompt_data.keys())}, "
                                f"출력 노드 수: {len(outputs)}"
                            )
                        else:
                            logger.warning(
                                f"프롬프트 ID {prompt_id}는 히스토리에 있지만 이미지가 없습니다. "
                                f"출력 구조: {list(prompt_data.keys())}, "
                                f"출력 노드 수: {len(outputs)}, "
                                f"상태: {status}"
                            )
                        # outputs 구조를 JSON으로 로깅 (디버깅용, 강화)
                        if outputs:
                            logger.debug("=" * 80)
                            logger.debug(f"[출력 구조 상세] 프롬프트 ID {prompt_id}의 모든 outputs:")
                            logger.debug(json.dumps(outputs, indent=2, ensure_ascii=False))
                            logger.debug("=" * 80)
                        
                        # status 상세 정보도 출력
                        if status:
                            logger.debug(f"[Status 상세] 프롬프트 ID {prompt_id}:")
                            logger.debug(json.dumps(status, indent=2, ensure_ascii=False))
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
                
            except RuntimeError as e:
                # 치명적 에러(RuntimeError)는 즉시 상위로 전파 (재시도하지 않음)
                error_msg = str(e)
                if "치명적 에러" in error_msg or "[Errno 22]" in error_msg:
                    # 치명적 에러는 재시도하지 않고 즉시 전파
                    raise
                # 치명적이지 않은 RuntimeError는 재시도 가능
                if attempt < max_retries - 1:
                    logger.warning(f"히스토리 API 조회 실패, {retry_delay}초 후 재시도 (시도 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    raise
            except requests.ConnectionError as e:
                error_msg = (
                    f"ComfyUI 서버({self.base_url})에 연결할 수 없습니다.\n"
                    f"서버를 수동으로 실행한 뒤 다시 시도해주세요."
                )
                if attempt < max_retries - 1:
                    logger.warning(f"히스토리 API 연결 실패, {retry_delay}초 후 재시도 (시도 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"히스토리 API 조회 실패, {retry_delay}초 후 재시도 (시도 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"히스토리 API 조회 최종 실패: {e}")
                    raise RuntimeError(
                        f"ComfyUI 히스토리 API 호출 실패 ({self.base_url}): {e}\n"
                        "서버가 실행 중이고 URL이 올바른지 확인해주세요."
                    ) from e
            except Exception as e:
                logger.error(f"히스토리 API 조회 중 예상치 못한 오류: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
        
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
            
            # 연결 실패 시 명확한 에러 메시지
            if response.status_code == 404:
                raise RuntimeError(
                    f"ComfyUI 서버({self.base_url})에서 이미지를 찾을 수 없습니다.\n"
                    f"이미지 파일명: {filename}\n"
                    "서버가 실행 중이고 이미지가 생성되었는지 확인해주세요."
                )
            
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"다운로드된 파일이 없습니다: {output_path}")
            
            logger.info(f"이미지 다운로드 완료: {output_path}")
            
        except requests.ConnectionError as e:
            error_msg = (
                f"ComfyUI 서버({self.base_url})에 연결할 수 없습니다.\n"
                f"서버를 수동으로 실행한 뒤 다시 시도해주세요.\n"
                f"이미지 파일명: {filename}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except requests.RequestException as e:
            logger.error(f"이미지 다운로드 실패 ({filename}): {e}")
            raise RuntimeError(
                f"이미지 다운로드 실패 ({self.base_url}): {e}\n"
                f"이미지 파일명: {filename}\n"
                "서버가 실행 중이고 이미지가 생성되었는지 확인해주세요."
            ) from e


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
    
    # ComfyUI 클라이언트 초기화 (서버는 외부에서 실행되어야 함)
    client = ComfyUIClient()
    
    # 서버 헬스체크 (선택적, 첫 번째 시도에서만)
    try:
        client.check_health()
        logger.debug("ComfyUI 서버 헬스체크 통과")
    except RuntimeError as e:
        logger.error(f"ComfyUI 서버 헬스체크 실패: {e}")
        raise
    
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
                # 프롬프트 실행 (매 재시도마다 새로운 prompt_id를 받음)
                prompt_id = client.execute_prompt(updated_workflow, retry_count=0)  # execute_prompt 내부에서 재시도
                
                # 완료 대기 (치명적 에러는 내부에서 즉시 예외 발생)
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
                
            except RuntimeError as e:
                # 서버 연결 실패 등 명확한 에러는 즉시 재시도하지 않고 상위로 전파
                error_msg = str(e)
                if "접속할 수 없습니다" in error_msg or "연결할 수 없습니다" in error_msg:
                    logger.error(f"ComfyUI 서버 연결 실패 (장면 {idx+1}): {e}")
                    raise
                
                # 치명적 에러인 경우 (OSError Errno 22 등) 재시도하지 않음
                if "치명적 에러" in error_msg or "[Errno 22]" in error_msg:
                    logger.error(f"치명적 에러 발생 (장면 {idx+1}): {e}")
                    raise
                
                last_error = e
                if attempt < retry_count:
                    wait_time = (attempt + 1) * 3
                    logger.warning(f"이미지 생성 실패 (장면 {idx+1}, 시도 {attempt + 1}/{retry_count + 1}), {wait_time}초 후 재시도: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"이미지 생성 최종 실패 (장면 {idx+1}): {e}")
                    raise
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
        
        topic = state.get("topic", "")
        output_dir = Config.get_output_dir(topic=topic)
        updated_scenes = generate_images(scenes, output_dir)
        
        return {
            **state,
            "scenes": updated_scenes
        }
        
    except RuntimeError as e:
        # 서버 연결 실패 등 명확한 에러는 사용자 친화적 메시지로 전달
        error_msg = str(e)
        if "접속할 수 없습니다" in error_msg or "연결할 수 없습니다" in error_msg:
            logger.error(f"[이미지 생성 노드] ComfyUI 서버 연결 실패: {e}")
            raise RuntimeError(
                f"ComfyUI 서버에 연결할 수 없습니다.\n"
                f"서버를 수동으로 실행한 뒤 다시 시도해주세요.\n"
                f"상세: {error_msg}"
            ) from e
        raise
    except Exception as e:
        logger.error(f"[이미지 생성 노드] 오류: {e}", exc_info=True)
        raise


# ============================================================================
# CLI 엔트리 포인트 (단일 프롬프트 테스트용)
# ============================================================================

def main_cli():
    """
    단일 프롬프트 테스트용 CLI 엔트리 포인트
    
    사용법:
        python -m src.tools.comfyui_client --prompt "test girl, anime, 4k" --out test.png
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ComfyUI 클라이언트 테스트 도구 (단일 프롬프트 실행)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m src.tools.comfyui_client --prompt "test girl, anime, 4k" --out test.png
  python -m src.tools.comfyui_client --prompt "landscape" --out output.png --workflow custom_workflow.json
        """
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="이미지 생성 프롬프트"
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        default="test_output.png",
        help="출력 이미지 파일 경로 (기본값: test_output.png)"
    )
    parser.add_argument(
        "--workflow", "-w",
        type=str,
        default=None,
        help="워크플로우 JSON 파일 경로 (기본값: COMFYUI_WORKFLOW_PATH 환경변수 또는 comfyui_workflow.json)"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=None,
        help="타임아웃 (초, 기본값: COMFYUI_TIMEOUT 환경변수 또는 300)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 로그 출력"
    )
    
    args = parser.parse_args()
    
    # 로깅 설정 (CLI 시작 시점에 명시적으로 설정)
    import sys
    from pathlib import Path
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    
    if args.verbose:
        # verbose 모드: 모든 로그 DEBUG 레벨
        root_logger.setLevel(logging.DEBUG)
        # 모든 핸들러의 레벨도 DEBUG로 설정
        for handler in root_logger.handlers:
            handler.setLevel(logging.DEBUG)
        # 모든 로거에 대해 DEBUG 레벨 설정
        for logger_name in logging.Logger.manager.loggerDict:
            log = logging.getLogger(logger_name)
            log.setLevel(logging.DEBUG)
            for handler in log.handlers:
                handler.setLevel(logging.DEBUG)
    else:
        # 기본 모드: 루트는 INFO, comfyui_client는 DEBUG
        root_logger.setLevel(logging.INFO)
        
        # comfyui_client 모듈 로거 명시적으로 설정
        comfyui_logger = logging.getLogger("src.tools.comfyui_client")
        comfyui_logger.setLevel(logging.DEBUG)
        comfyui_logger.propagate = True  # 루트 로거로 전파
        
        # comfyui_client 로거에 DEBUG 레벨 핸들러 추가 (없는 경우)
        if not comfyui_logger.handlers:
            # 콘솔 핸들러
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            comfyui_logger.addHandler(console_handler)
            
            # 파일 핸들러 (logs 디렉토리에)
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "comfyui_client_debug.log", encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            comfyui_logger.addHandler(file_handler)
        else:
            # 기존 핸들러의 레벨도 DEBUG로 설정
            for handler in comfyui_logger.handlers:
                handler.setLevel(logging.DEBUG)
    
    try:
        # ComfyUI 클라이언트 초기화
        client = ComfyUIClient()
        
        # 헬스체크
        print(f"ComfyUI 서버 연결 확인 중... ({client.base_url})")
        client.check_health()
        print("✓ ComfyUI 서버 연결 성공")
        
        # 워크플로우 로드
        workflow_path = args.workflow or Config.COMFYUI_WORKFLOW_PATH
        print(f"워크플로우 로드 중... ({workflow_path})")
        workflow = client.load_workflow(workflow_path)
        print("✓ 워크플로우 로드 완료")
        
        # 프롬프트 업데이트
        print(f"프롬프트 업데이트 중... ({args.prompt[:50]}...)")
        updated_workflow = client.update_prompt(workflow, args.prompt)
        print("✓ 프롬프트 업데이트 완료")
        
        # 프롬프트 실행
        print("ComfyUI에 프롬프트 실행 요청 중...")
        prompt_id = client.execute_prompt(updated_workflow)
        print(f"✓ 프롬프트 실행됨 (ID: {prompt_id})")
        print(f"  프롬프트 워크플로우 저장됨: logs/comfyui_prompts/{prompt_id}.json")
        
        # 완료 대기
        timeout = args.timeout or Config.COMFYUI_TIMEOUT
        print(f"이미지 생성 완료 대기 중... (타임아웃: {timeout}초)")
        image_filenames = client.wait_for_completion(prompt_id, timeout=timeout)
        print(f"✓ 이미지 생성 완료 ({len(image_filenames)}개 파일)")
        
        # 이미지 다운로드
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"이미지 다운로드 중... ({image_filenames[0]})")
        client.download_result(image_filenames[0], str(output_path))
        print(f"✓ 이미지 다운로드 완료: {output_path}")
        
        print("\n" + "=" * 60)
        print("성공!")
        print("=" * 60)
        print(f"출력 파일: {output_path.absolute()}")
        print(f"프롬프트 ID: {prompt_id}")
        print(f"프롬프트 워크플로우: logs/comfyui_prompts/{prompt_id}.json")
        if Path(f"logs/comfyui_history/{prompt_id}.json").exists():
            print(f"ComfyUI History: logs/comfyui_history/{prompt_id}.json")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
        return 130
        
    except Exception as e:
        print(f"\n오류 발생: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main_cli())

