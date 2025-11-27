"""
ComfyUI 워크플로우 자동 수정 스크립트
ckpt_name 자동 정정 및 출력 인덱스 교정
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from src.config import Config


def find_checkpoint_node(workflow: Dict) -> Optional[Tuple[str, Dict]]:
    """
    CheckpointLoaderSimple 노드를 찾습니다.
    
    Returns:
        (node_id, node_data) 튜플 또는 None
    """
    for node_id, node_data in workflow.items():
        if node_data.get("class_type") == "CheckpointLoaderSimple":
            return (node_id, node_data)
    return None


def get_checkpoint_from_env() -> Optional[str]:
    """환경 변수에서 체크포인트 이름을 가져옵니다."""
    checkpoint = os.getenv("COMFYUI_CHECKPOINT")
    if checkpoint:
        return checkpoint.strip()
    return None


def scan_checkpoint_directories() -> List[str]:
    """
    체크포인트 디렉토리를 스캔하여 .safetensors 파일을 찾습니다.
    
    Returns:
        찾은 체크포인트 파일명 리스트
    """
    checkpoints = []
    search_paths = [
        Path("models/checkpoints"),
        Path("ComfyUI/models/checkpoints"),
        Path.home() / "ComfyUI/models/checkpoints",
    ]
    
    for search_path in search_paths:
        if search_path.exists() and search_path.is_dir():
            for file in search_path.glob("*.safetensors"):
                checkpoints.append(file.name)
                print(f"  발견: {search_path / file.name}")
    
    return checkpoints


def get_available_models_from_api() -> List[str]:
    """
    ComfyUI API에서 사용 가능한 모델 목록을 가져옵니다.
    
    Returns:
        사용 가능한 모델 파일명 리스트
    """
    try:
        url = f"{Config.COMFYUI_URL}/object_info"
        headers = {}
        if Config.COMFYUI_API_KEY:
            headers["Authorization"] = f"Bearer {Config.COMFYUI_API_KEY}"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        object_info = response.json()
        checkpoint_info = object_info.get("CheckpointLoaderSimple", {})
        input_info = checkpoint_info.get("input", {})
        required = input_info.get("required", {})
        ckpt_name_info = required.get("ckpt_name", [])
        
        if isinstance(ckpt_name_info, list) and len(ckpt_name_info) > 0:
            models = ckpt_name_info[0] if isinstance(ckpt_name_info[0], list) else ckpt_name_info
            return models
    except Exception as e:
        print(f"  API에서 모델 목록 조회 실패: {e}")
    
    return []


def select_checkpoint_name(workflow: Dict, checkpoint_node_id: str) -> str:
    """
    체크포인트 이름을 자동으로 선택합니다.
    
    우선순위:
    1. .env 파일의 COMFYUI_CHECKPOINT
    2. ComfyUI API에서 가져온 목록의 첫 번째
    3. 로컬 디렉토리 스캔 결과의 첫 번째
    4. 기본값 (flux1-dev.safetensors 또는 sdxl.safetensors)
    """
    print("\n[1단계] 체크포인트 이름 자동 선택 중...")
    
    # 1. 환경 변수 확인
    env_checkpoint = get_checkpoint_from_env()
    if env_checkpoint:
        print(f"  ✓ .env 파일에서 발견: {env_checkpoint}")
        return env_checkpoint
    
    # 2. ComfyUI API에서 가져오기
    print("  ComfyUI API에서 모델 목록 조회 중...")
    api_models = get_available_models_from_api()
    if api_models:
        selected = api_models[0]
        print(f"  ✓ API에서 발견: {selected} (총 {len(api_models)}개 모델)")
        return selected
    
    # 3. 로컬 디렉토리 스캔
    print("  로컬 디렉토리 스캔 중...")
    local_checkpoints = scan_checkpoint_directories()
    if local_checkpoints:
        selected = local_checkpoints[0]
        print(f"  ✓ 로컬에서 발견: {selected} (총 {len(local_checkpoints)}개 파일)")
        return selected
    
    # 4. 기본값
    default_models = ["flux1-dev.safetensors", "sdxl.safetensors", "v1-5-pruned-emaonly.safetensors"]
    for default in default_models:
        print(f"  ⚠ 기본값 사용: {default}")
        return default
    
    # 최후의 수단
    print("  ⚠ 경고: 체크포인트를 찾을 수 없어 기본값 사용")
    return "flux1-dev.safetensors"


def fix_checkpoint_connections(workflow: Dict, checkpoint_node_id: str) -> Dict:
    """
    CheckpointLoaderSimple의 출력 인덱스를 자동으로 교정합니다.
    
    출력 인덱스:
    - [0] MODEL
    - [1] CLIP
    - [2] VAE
    
    연결 규칙:
    - KSampler의 model → ["<CheckpointNodeId>", 0]
    - CLIPTextEncode의 clip → ["<CheckpointNodeId>", 1]
    - VAEDecode의 vae → ["<CheckpointNodeId>", 2]
    """
    print(f"\n[2단계] CheckpointLoaderSimple 출력 인덱스 교정 중... (노드 ID: {checkpoint_node_id})")
    
    fixed_count = 0
    
    for node_id, node_data in workflow.items():
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})
        
        # KSampler의 model 입력 수정
        if class_type == "KSampler" and "model" in inputs:
            current = inputs["model"]
            expected = [checkpoint_node_id, 0]
            if current != expected:
                workflow[node_id]["inputs"]["model"] = expected
                print(f"  ✓ KSampler 노드 {node_id}: model 연결 수정 [{current[0]}, {current[1]}] → [{expected[0]}, {expected[1]}]")
                fixed_count += 1
        
        # CLIPTextEncode의 clip 입력 수정
        if class_type == "CLIPTextEncode" and "clip" in inputs:
            current = inputs["clip"]
            expected = [checkpoint_node_id, 1]
            if current != expected:
                workflow[node_id]["inputs"]["clip"] = expected
                print(f"  ✓ CLIPTextEncode 노드 {node_id}: clip 연결 수정 [{current[0]}, {current[1]}] → [{expected[0]}, {expected[1]}]")
                fixed_count += 1
        
        # VAEDecode의 vae 입력 수정
        if class_type == "VAEDecode" and "vae" in inputs:
            current = inputs["vae"]
            expected = [checkpoint_node_id, 2]
            if current != expected:
                workflow[node_id]["inputs"]["vae"] = expected
                print(f"  ✓ VAEDecode 노드 {node_id}: vae 연결 수정 [{current[0]}, {current[1]}] → [{expected[0]}, {expected[1]}]")
                fixed_count += 1
    
    if fixed_count == 0:
        print("  ✓ 모든 연결이 올바릅니다.")
    else:
        print(f"  ✓ 총 {fixed_count}개 연결 수정 완료")
    
    return workflow


def validate_workflow(workflow: Dict, checkpoint_node_id: str) -> bool:
    """
    워크플로우의 타입 오류를 검사합니다.
    
    Returns:
        검증 통과 여부
    """
    print(f"\n[3단계] 타입 오류 검사 중...")
    
    errors = []
    
    for node_id, node_data in workflow.items():
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})
        
        # CLIPTextEncode의 clip 입력 검사
        if class_type == "CLIPTextEncode" and "clip" in inputs:
            clip_input = inputs["clip"]
            if clip_input[0] == checkpoint_node_id and clip_input[1] != 1:
                errors.append(f"노드 {node_id} (CLIPTextEncode): clip 입력이 올바른 인덱스가 아닙니다. [{clip_input[0]}, {clip_input[1]}] → [{checkpoint_node_id}, 1]")
        
        # VAEDecode의 vae 입력 검사
        if class_type == "VAEDecode" and "vae" in inputs:
            vae_input = inputs["vae"]
            if vae_input[0] == checkpoint_node_id and vae_input[1] != 2:
                errors.append(f"노드 {node_id} (VAEDecode): vae 입력이 올바른 인덱스가 아닙니다. [{vae_input[0]}, {vae_input[1]}] → [{checkpoint_node_id}, 2]")
        
        # KSampler의 model 입력 검사
        if class_type == "KSampler" and "model" in inputs:
            model_input = inputs["model"]
            if model_input[0] == checkpoint_node_id and model_input[1] != 0:
                errors.append(f"노드 {node_id} (KSampler): model 입력이 올바른 인덱스가 아닙니다. [{model_input[0]}, {model_input[1]}] → [{checkpoint_node_id}, 0]")
    
    if errors:
        print(f"  ⚠ {len(errors)}개 오류 발견:")
        for error in errors:
            print(f"    - {error}")
        return False
    else:
        print("  ✓ 타입 검사 통과")
        return True


def fix_workflow(input_path: str = "comfyui_workflow.json", output_path: str = "comfyui_workflow_fixed.json") -> Dict:
    """
    ComfyUI 워크플로우를 자동으로 수정합니다.
    
    Args:
        input_path: 입력 워크플로우 파일 경로
        output_path: 출력 워크플로우 파일 경로
        
    Returns:
        수정된 워크플로우 딕셔너리
    """
    print("=" * 60)
    print("ComfyUI 워크플로우 자동 수정 시작")
    print("=" * 60)
    
    # 워크플로우 로드
    print(f"\n워크플로우 로드: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    # CheckpointLoaderSimple 노드 찾기
    checkpoint_info = find_checkpoint_node(workflow)
    if not checkpoint_info:
        raise ValueError("CheckpointLoaderSimple 노드를 찾을 수 없습니다.")
    
    checkpoint_node_id, checkpoint_node = checkpoint_info
    print(f"CheckpointLoaderSimple 노드 발견: {checkpoint_node_id}")
    
    # 1. ckpt_name 자동 정정
    selected_checkpoint = select_checkpoint_name(workflow, checkpoint_node_id)
    workflow[checkpoint_node_id]["inputs"]["ckpt_name"] = selected_checkpoint
    print(f"  → 최종 선택: {selected_checkpoint}")
    
    # 2. 출력 인덱스 자동 교정
    workflow = fix_checkpoint_connections(workflow, checkpoint_node_id)
    
    # 3. 타입 오류 검사 및 재수정
    max_retries = 3
    for attempt in range(max_retries):
        if validate_workflow(workflow, checkpoint_node_id):
            break
        else:
            if attempt < max_retries - 1:
                print(f"  재수정 시도 {attempt + 1}/{max_retries - 1}...")
                workflow = fix_checkpoint_connections(workflow, checkpoint_node_id)
    
    # 수정된 워크플로우 저장
    print(f"\n[4단계] 수정된 워크플로우 저장: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("수정 완료!")
    print("=" * 60)
    
    return workflow


if __name__ == "__main__":
    try:
        fixed_workflow = fix_workflow()
        
        print("\n[수정 사항 요약]")
        print("1. ✓ ckpt_name 자동 정정 완료")
        print("2. ✓ CheckpointLoaderSimple 출력 인덱스 교정 완료")
        print("3. ✓ 타입 오류 검사 및 복구 완료")
        print("4. ✓ comfyui_workflow_fixed.json 저장 완료")
        
        print("\n[수정된 워크플로우]")
        print(json.dumps(fixed_workflow, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

