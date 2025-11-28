"""
Wan2.2 I2V 모션 생성 모듈
ComfyUI의 Wan2.2 Distill I2V 모델을 사용하여 정적 이미지를 애니메이션 프레임 시퀀스로 변환합니다.
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
        
        Args:
            client: ComfyUI 클라이언트 (None이면 새로 생성)
        """
        self.client = client or ComfyUIClient()
        self.motion_fps = Config.MOTION_FPS
        self.motion_duration = Config.MOTION_DURATION
        self.frames_per_scene = int(self.motion_fps * self.motion_duration)
        self.i2v_steps = Config.I2V_STEPS
        self.i2v_guidance = Config.I2V_GUIDANCE
        self.i2v_checkpoint = Config.I2V_CHECKPOINT
        self.i2v_lora = Config.I2V_LORA
        self.i2v_node_type = Config.I2V_NODE_TYPE
    
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
    
    def create_wan2_i2v_workflow(self, image_filename: str, width: int, height: int, seed: int = None) -> Dict:
        """
        Wan2.2 I2V 워크플로우를 생성합니다.
        
        Args:
            image_filename: 입력 이미지 파일명 (ComfyUI에서 접근 가능한 이름)
            width: 이미지 너비
            height: 이미지 높이
            seed: 시드값 (None이면 랜덤)
            
        Returns:
            Wan2.2 I2V 워크플로우 딕셔너리
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Wan2.2 I2V 워크플로우 구조
        # 노드 구조:
        # 1. LoadImage - 입력 이미지 로드
        # 2. CheckpointLoaderSimple - Wan2.2-Distill-Loras 체크포인트
        # 3. LoraLoader - Wan2.2 I2V Distill LoRA
        # 4. I2V 노드 (Wan2.2 전용) - steps, guidance, fps, motion_length 설정
        # 5. SaveImage - 프레임 저장
        
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
                    "ckpt_name": self.i2v_checkpoint
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint (Wan2.2-Distill-Loras)"}
            },
            "3": {
                "inputs": {
                    "model": ["2", 0],
                    "clip": ["2", 1],
                    "lora_name": self.i2v_lora,
                    "strength_model": 1.0,
                    "strength_clip": 1.0
                },
                "class_type": "LoraLoader",
                "_meta": {"title": "Load LoRA (Wan2.2 I2V Distill)"}
            },
            "4": {
                "inputs": {
                    "image": ["1", 0],
                    "model": ["3", 0],
                    "clip": ["3", 1],
                    "vae": ["2", 2],
                    "steps": self.i2v_steps,
                    "guidance": self.i2v_guidance,
                    "fps": self.motion_fps,
                    "motion_length": int(self.motion_duration),
                    "width": width,
                    "height": height,
                    "seed": seed
                },
                "class_type": self.i2v_node_type,
                "_meta": {"title": "Wan2.2 I2V Generation"}
            },
            "5": {
                "inputs": {
                    "filename_prefix": "wan2_i2v",
                    "images": ["4", 0]
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
                response.raise_for_status()
                
                result = response.json()
                # ComfyUI는 보통 {"name": "filename.png"} 형식으로 반환
                uploaded_filename = result.get("name") or os.path.basename(image_path)
                logger.info(f"[모션 생성] 이미지 업로드 완료: {uploaded_filename}")
                return uploaded_filename
                
        except Exception as e:
            logger.error(f"[모션 생성] 이미지 업로드 실패: {e}")
            # 업로드 실패 시 파일명만 반환 (이미 output 폴더에 있을 수 있음)
            return os.path.basename(image_path)
    
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
            workflow = self.create_wan2_i2v_workflow(image_filename, width, height, seed)
            
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
        
    except Exception as e:
        logger.error(f"[모션 생성 노드] 오류: {e}", exc_info=True)
        raise
