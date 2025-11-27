"""
ComfyUI 모의(Mock) 모듈
ComfyUI 서버 없이 테스트하기 위한 더미 이미지 생성 기능
"""

import logging
from pathlib import Path
from typing import Dict, List
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)


def generate_mock_image(image_prompt: str, output_path: Path, width: int = 1024, height: int = 1024) -> None:
    """
    더미 이미지를 생성합니다 (ComfyUI 없이 테스트용).
    
    Args:
        image_prompt: 이미지 프롬프트 (텍스트로 표시)
        output_path: 저장할 경로
        width: 이미지 너비
        height: 이미지 높이
    """
    # 그라데이션 배경 생성
    img = Image.new('RGB', (width, height), color='#2C3E50')
    draw = ImageDraw.Draw(img)
    
    # 그라데이션 효과
    for y in range(height):
        r = int(44 + (y / height) * 20)  # 44-64
        g = int(62 + (y / height) * 30)  # 62-92
        b = int(80 + (y / height) * 40)  # 80-120
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # 프롬프트 텍스트 추가
    try:
        # 시스템 폰트 사용 시도
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
        except:
            font = ImageFont.load_default()
    
    # 텍스트를 여러 줄로 나누기
    words = image_prompt.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width < width - 100:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # 텍스트 그리기
    text_y = height // 2 - (len(lines) * 50) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (width - text_width) // 2
        
        # 텍스트 그림자
        draw.text((text_x + 2, text_y + 2), line, font=font, fill='#000000')
        # 텍스트
        draw.text((text_x, text_y), line, font=font, fill='#FFFFFF')
        text_y += 50
    
    # "MOCK IMAGE" 라벨 추가
    label = "MOCK IMAGE (ComfyUI not available)"
    try:
        label_font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
        except:
            label_font = ImageFont.load_default()
    
    label_bbox = draw.textbbox((0, 0), label, font=label_font)
    label_width = label_bbox[2] - label_bbox[0]
    label_x = (width - label_width) // 2
    draw.text((label_x, height - 40), label, font=label_font, fill='#FFA500')
    
    # 이미지 저장
    img.save(output_path, 'PNG')
    logger.info(f"더미 이미지 생성 완료: {output_path}")


def generate_images_mock(scenes: List[Dict], output_dir: Path) -> List[Dict]:
    """
    모든 장면의 더미 이미지를 생성합니다 (ComfyUI 없이 테스트용).
    
    Args:
        scenes: 장면 리스트
        output_dir: 출력 디렉토리
        
    Returns:
        이미지 경로가 업데이트된 장면 리스트
    """
    logger.warning("[이미지 생성] ComfyUI 모의 모드: 더미 이미지 생성 중...")
    logger.warning("[이미지 생성] 실제 ComfyUI 서버를 사용하려면 COMFYUI_USE_MOCK=false로 설정하세요.")
    
    updated_scenes = []
    
    for idx, scene in enumerate(scenes):
        image_prompt = scene.get('image_prompt', '').strip()
        if not image_prompt:
            raise ValueError(f"장면 {idx+1}에 이미지 프롬프트가 없습니다.")
        
        logger.info(f"[이미지 생성] 장면 {idx+1} 더미 이미지 생성 중: {image_prompt[:50]}...")
        
        image_path = output_dir / f"image_scene_{idx+1}.png"
        generate_mock_image(image_prompt, image_path)
        
        scene['image_path'] = str(image_path)
        updated_scenes.append(scene)
        logger.info(f"[이미지 생성] 장면 {idx+1} 완료: {image_path}")
    
    logger.info(f"[이미지 생성] 모든 더미 이미지 생성 완료")
    return updated_scenes

