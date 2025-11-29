"""
Wan2.2 I2V 모션 생성 모듈

ComfyUI의 Wan2.2 Distill I2V 모델을 사용하여 정적 이미지를 애니메이션 프레임 시퀀스로 변환합니다.

중요:
- ComfyUI 서버는 외부에서 이미 실행되어야 합니다
- 이 모듈은 HTTP API만 사용하며, 서버 프로세스를 관리하지 않습니다
- ComfyUIClient를 내부적으로 사용하여 HTTP 요청을 보냅니다
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image

from src.config import Config
from src.tools.comfyui_client import ComfyUIClient

logger = logging.getLogger(__name__)


class MotionGenerator:
    """Wan2.2 I2V를 사용한 모션 생성 클래스"""
    
    def __init__(self, client: Optional[ComfyUIClient] = None):
        """
        모션 생성기 초기화
        
        ComfyUI 서버는 외부에서 이미 실행되어야 합니다.
        이 클래스는 서버를 실행하지 않고 HTTP API만 사용합니다.
        
        Args:
            client: ComfyUI 클라이언트 (None이면 새로 생성)
        """
        self.client = client or ComfyUIClient()
        
        # 서버 헬스체크 (선택적)
        try:
            self.client.check_health()
            logger.debug("ComfyUI 서버 헬스체크 통과 (모션 생성)")
        except RuntimeError as e:
            logger.warning(f"ComfyUI 서버 헬스체크 실패 (모션 생성): {e}")
            # 헬스체크 실패해도 계속 진행 (실제 API 호출 시 에러 발생)
        self.motion_fps = Config.MOTION_FPS
        self.motion_duration = Config.MOTION_DURATION
        self.frames_per_scene = int(self.motion_fps * self.motion_duration)
        self.i2v_steps = Config.I2V_STEPS
        self.i2v_guidance = Config.I2V_GUIDANCE
        self.i2v_node_type = Config.I2V_NODE_TYPE
        self.i2v_clip_name = Config.I2V_CLIP_NAME
        self.i2v_vae_name = Config.I2V_VAE_NAME
        self.i2v_unet_name = Config.I2V_UNET_NAME
        self.i2v_model_sampling_shift = Config.I2V_MODEL_SAMPLING_SHIFT
    
    def _get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """
        이미지의 해상도를 가져옵니다.
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            (width, height) 튜플
        """
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception as e:
            logger.warning(f"이미지 해상도 감지 실패, 기본값 사용: {e}")
            return (1024, 1024)  # 기본 해상도
    
    def create_wan2_i2v_workflow(self, image_filename: str, prompt: str, width: int, height: int, seed: int = None) -> Dict:
        """
        Wan2.2 I2V 워크플로우를 생성합니다.
        JSON 파일의 구조를 기반으로 올바른 워크플로우를 생성합니다.
        
        Args:
            image_filename: 입력 이미지 파일명 (ComfyUI에서 접근 가능한 이름)
            prompt: 모션 프롬프트
            width: 이미지 너비
            height: 이미지 높이
            seed: 시드값 (None이면 랜덤)
            
        Returns:
            Wan2.2 I2V 워크플로우 딕셔너리
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # 프레임 개수 계산 (length는 프레임 수)
        length = self.frames_per_scene
        
        # Negative prompt (기본값)
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        
        # Wan2.2 I2V 워크플로우 구조 (JSON 기반):
        # 1. LoadImage - 입력 이미지 로드
        # 2. CLIPLoader - CLIP 텍스트 인코더 로드
        # 3. CLIPTextEncode - 프롬프트 인코딩 (positive, negative)
        # 4. VAELoader - VAE 로드
        # 5. UNETLoader - UNet 모델 로드 (high_noise)
        # 6. ModelSamplingSD3 - 모델 샘플링 설정
        # 7. WanImageToVideo - I2V 변환 (CONDITIONING, LATENT 출력)
        # 8. KSamplerAdvanced - 샘플링
        # 9. VAEDecode - 디코딩
        # 10. SaveImage - 프레임 저장
        
        workflow = {
            "1": {
                "inputs": {
                    "image": image_filename
                },
                "class_type": "LoadImage",
                "_meta": {"title": "Load Image"}
            },
            "2": {
                "inputs": {
                    "clip_name": self.i2v_clip_name,
                    "type": "wan",
                    "device": "default"
                },
                "class_type": "CLIPLoader",
                "_meta": {"title": "Load CLIP"}
            },
            "3": {
                "inputs": {
                    "clip": ["2", 0],
                    "text": prompt
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
            },
            "4": {
                "inputs": {
                    "clip": ["2", 0],
                    "text": negative_prompt
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Negative Prompt)"}
            },
            "5": {
                "inputs": {
                    "vae_name": self.i2v_vae_name
                },
                "class_type": "VAELoader",
                "_meta": {"title": "Load VAE"}
            },
            "6": {
                "inputs": {
                    "unet_name": self.i2v_unet_name,
                    "weight_dtype": "default"
                },
                "class_type": "UNETLoader",
                "_meta": {"title": "Load UNet (High Noise)"}
            },
            "7": {
                "inputs": {
                    "model": ["6", 0],
                    "shift": self.i2v_model_sampling_shift
                },
                "class_type": "ModelSamplingSD3",
                "_meta": {"title": "Model Sampling SD3"}
            },
            "8": {
                "inputs": {
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "vae": ["5", 0],
                    "clip_vision_output": None,
                    "start_image": ["1", 0],
                    "width": width,
                    "height": height,
                    "length": length,
                    "batch_size": 1
                },
                "class_type": "WanImageToVideo",
                "_meta": {"title": "Wan Image to Video"}
            },
            "9": {
                "inputs": {
                    "model": ["7", 0],
                    "positive": ["8", 0],
                    "negative": ["8", 1],
                    "latent_image": ["8", 2],
                    "add_noise": "enable",
                    "noise_seed": seed,
                    "steps": self.i2v_steps,
                    "cfg": self.i2v_guidance,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "start_at_step": 0,
                    "end_at_step": 10,
                    "return_with_leftover_noise": "enable"
                },
                "class_type": "KSamplerAdvanced",
                "_meta": {"title": "KSampler Advanced"}
            },
            "10": {
                "inputs": {
                    "samples": ["9", 0],
                    "vae": ["5", 0]
                },
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"}
            },
            "11": {
                "inputs": {
                    "filename_prefix": "wan2_i2v",
                    "images": ["10", 0]
                },
                "class_type": "SaveImage",
                "_meta": {"title": "Save Image Frames"}
            }
        }
        
        return workflow
    
    def _upload_image(self, image_path: str) -> str:
        """
        이미지를 ComfyUI에 업로드합니다.
        
        Args:
            image_path: 업로드할 이미지 파일 경로
            
        Returns:
            업로드된 이미지 파일명
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")
        
        upload_url = f"{self.client.base_url}/upload/image"
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (os.path.basename(image_path), f, 'image/png')}
                response = requests.post(upload_url, files=files, timeout=60)
                
                # 연결 실패 시 명확한 에러 메시지
                if response.status_code == 404:
                    raise RuntimeError(
                        f"ComfyUI 서버({self.client.base_url})를 찾을 수 없습니다.\n"
                        "서버가 실행 중인지 확인해주세요."
                    )
                
                response.raise_for_status()
                
                result = response.json()
                # ComfyUI는 보통 {"name": "filename.png"} 형식으로 반환
                uploaded_filename = result.get("name") or os.path.basename(image_path)
                logger.info(f"[모션 생성] 이미지 업로드 완료: {uploaded_filename}")
                return uploaded_filename
                
        except requests.ConnectionError as e:
            error_msg = (
                f"ComfyUI 서버({self.client.base_url})에 연결할 수 없습니다.\n"
                f"서버를 수동으로 실행한 뒤 다시 시도해주세요.\n"
                f"이미지 파일: {image_path}"
            )
            logger.error(f"[모션 생성] 이미지 업로드 실패: {error_msg}")
            raise RuntimeError(error_msg) from e
        except requests.RequestException as e:
            error_msg = (
                f"이미지 업로드 실패 ({self.client.base_url}): {e}\n"
                f"이미지 파일: {image_path}\n"
                "서버가 실행 중이고 URL이 올바른지 확인해주세요."
            )
            logger.error(f"[모션 생성] {error_msg}")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            logger.error(f"[모션 생성] 이미지 업로드 실패: {e}")
            raise RuntimeError(
                f"이미지 업로드 중 예상치 못한 오류: {e}\n"
                f"이미지 파일: {image_path}"
            ) from e
    
    def generate_motion_frames(self, image_path: str, prompt: str, output_dir: Path, scene_index: int) -> List[str]:
        """
        정적 이미지를 애니메이션 프레임 시퀀스로 변환합니다.
        
        Args:
            image_path: 입력 이미지 파일 경로
            prompt: 모션 프롬프트 (Wan2.2 I2V는 프롬프트를 사용하지 않지만 호환성을 위해 유지)
            output_dir: 출력 디렉토리
            scene_index: 장면 인덱스
            
        Returns:
            생성된 프레임 파일 경로 리스트
        """
        logger.info(f"[모션 생성] 장면 {scene_index + 1} 시작: {image_path}")
        
        # 프레임 저장 디렉토리 생성
        frames_dir = output_dir / "motion" / f"scene_{scene_index + 1}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 이미지 해상도 감지
            width, height = self._get_image_dimensions(image_path)
            logger.info(f"[모션 생성] 이미지 해상도: {width}x{height}")
            
            # 이미지 업로드 (ComfyUI가 접근할 수 있도록)
            image_filename = self._upload_image(image_path)
            
            # I2V 노드 존재 확인
            if not self.client.check_node_exists(self.i2v_node_type):
                raise ValueError(
                    f"I2V 노드 '{self.i2v_node_type}'가 ComfyUI에 설치되지 않았습니다. "
                    f"환경 변수 I2V_NODE_TYPE을 실제 노드 이름으로 설정하거나, "
                    f"ComfyUI에 해당 노드를 설치해주세요."
                )
            
            # Wan2.2 I2V 워크플로우 생성
            seed = random.randint(0, 2**32 - 1)
            workflow = self.create_wan2_i2v_workflow(image_filename, prompt, width, height, seed)
            
            logger.info(
                f"[모션 생성] Wan2.2 I2V 워크플로우 생성 완료 "
                f"(노드: {self.i2v_node_type}, 시드: {seed}, 프레임: {self.frames_per_scene}, "
                f"스텝: {self.i2v_steps}, Guidance: {self.i2v_guidance})"
            )
            logger.debug(f"[모션 생성] 워크플로우 노드 수: {len(workflow)}")
            logger.debug(f"[모션 생성] 워크플로우 JSON: {json.dumps(workflow, indent=2, ensure_ascii=False)}")
            
            # ComfyUI에 프롬프트 실행
            prompt_id = self.client.execute_prompt(workflow, retry_count=0)
            logger.info(f"[모션 생성] 프롬프트 실행됨 (ID: {prompt_id})")
            
            # 완료 대기 및 프레임 다운로드
            image_filenames = self.client.wait_for_completion(prompt_id)
            
            if not image_filenames:
                raise ValueError(f"장면 {scene_index + 1}의 모션 프레임이 생성되지 않았습니다.")
            
            logger.info(f"[모션 생성] {len(image_filenames)}개 프레임 생성됨")
            
            # 예상 프레임 개수 검증
            expected_frames = self.frames_per_scene
            if len(image_filenames) != expected_frames:
                logger.warning(
                    f"[모션 생성] 프레임 개수 불일치: "
                    f"예상 {expected_frames}개, 실제 {len(image_filenames)}개"
                )
            
            # 프레임 다운로드 및 저장
            frame_paths = []
            for idx, filename in enumerate(image_filenames):
                frame_path = frames_dir / f"frame_{idx:04d}.png"
                try:
                    self.client.download_result(filename, str(frame_path))
                    frame_paths.append(str(frame_path))
                    logger.debug(f"[모션 생성] 프레임 {idx + 1}/{len(image_filenames)} 다운로드 완료")
                except Exception as e:
                    logger.warning(f"[모션 생성] 프레임 {idx + 1} 다운로드 실패: {e}")
            
            if not frame_paths:
                raise ValueError(f"장면 {scene_index + 1}의 프레임을 다운로드할 수 없습니다.")
            
            # 최종 프레임 개수 검증
            logger.info(
                f"[모션 생성] 장면 {scene_index + 1} 완료: "
                f"{len(frame_paths)}개 프레임 저장됨 "
                f"(예상: {expected_frames}개, FPS: {self.motion_fps}, "
                f"길이: {self.motion_duration}초)"
            )
            
            return frame_paths
            
        except Exception as e:
            logger.error(f"[모션 생성] 장면 {scene_index + 1} 실패: {e}", exc_info=True)
            raise
    
    def generate_all_motions(self, scenes: List[Dict], output_dir: Path) -> List[Dict]:
        """
        모든 장면의 모션 프레임을 생성합니다.
        
        Args:
            scenes: 장면 리스트 (image_path 포함)
            output_dir: 출력 디렉토리
            
        Returns:
            motion_frames_path가 추가된 장면 리스트
        """
        logger.info(f"[모션 생성] {len(scenes)}개 장면의 모션 생성 시작 (Wan2.2 I2V)")
        logger.info(
            f"[모션 생성] 설정: FPS={self.motion_fps}, "
            f"길이={self.motion_duration}초, "
            f"프레임/장면={self.frames_per_scene}, "
            f"스텝={self.i2v_steps}, Guidance={self.i2v_guidance}"
        )
        
        updated_scenes = []
        
        for idx, scene in enumerate(scenes):
            image_path = scene.get('image_path', '')
            image_prompt = scene.get('image_prompt', '')
            
            if not image_path:
                raise ValueError(f"장면 {idx+1}에 이미지 경로가 없습니다.")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")
            
            # 모션 프레임 생성
            frame_paths = self.generate_motion_frames(
                image_path=image_path,
                prompt=image_prompt,
                output_dir=output_dir,
                scene_index=idx
            )
            
            # 프레임 개수 최종 검증
            if len(frame_paths) != self.frames_per_scene:
                logger.warning(
                    f"[모션 생성] 장면 {idx+1}: 프레임 개수 불일치 "
                    f"(예상: {self.frames_per_scene}, 실제: {len(frame_paths)})"
                )
            
            # 장면에 모션 프레임 경로 추가
            scene['motion_frames_path'] = frame_paths
            scene['motion_frames_dir'] = str(Path(frame_paths[0]).parent) if frame_paths else ""
            updated_scenes.append(scene)
            
            logger.info(f"[모션 생성] 장면 {idx+1} 완료: {len(frame_paths)}개 프레임")
        
        logger.info(f"[모션 생성] 모든 모션 생성 완료")
        return updated_scenes


def node_motion_generator(state: Dict) -> Dict:
    """
    LangGraph 노드 함수: 모션 생성 (Wan2.2 I2V)
    
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
        
        # 모션 생성 비활성화 옵션 확인
        if Config.MOTION_ENABLED:
            generator = MotionGenerator()
            updated_scenes = generator.generate_all_motions(scenes, output_dir)
        else:
            logger.info("[모션 생성] 모션 생성이 비활성화되어 있습니다. 스킵합니다.")
            updated_scenes = scenes
        
        return {
            **state,
            "scenes": updated_scenes
        }
        
    except RuntimeError as e:
        # 서버 연결 실패 등 명확한 에러는 사용자 친화적 메시지로 전달
        error_msg = str(e)
        if "접속할 수 없습니다" in error_msg or "연결할 수 없습니다" in error_msg:
            logger.error(f"[모션 생성 노드] ComfyUI 서버 연결 실패: {e}")
            raise RuntimeError(
                f"ComfyUI 서버에 연결할 수 없습니다.\n"
                f"서버를 수동으로 실행한 뒤 다시 시도해주세요.\n"
                f"상세: {error_msg}"
            ) from e
        raise
    except Exception as e:
        logger.error(f"[모션 생성 노드] 오류: {e}", exc_info=True)
        raise
