"""
LangGraph 워크플로우 정의 모듈 (메인 오케스트레이터)
상태 정의, 노드 등록, 그래프 구성을 담당합니다.

이 모듈은 메인 오케스트레이터 그래프를 정의하며,
3개의 서브그래프를 노드처럼 호출합니다:
- Script Pipeline 서브그래프
- Asset Generation 서브그래프
- Video Assembly 서브그래프
"""

import logging
from typing import TypedDict, List, Dict, Annotated
from functools import wraps

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.utils.cache import get_cache, WorkflowCache
from src.workflow.script_graph import build_script_graph
from src.workflow.asset_graph import build_asset_graph
from src.workflow.video_graph import build_video_graph

logger = logging.getLogger(__name__)


# ============================================================================
# 상태 정의
# ============================================================================

class VideoState(TypedDict):
    """
    비디오 생성 워크플로우의 상태를 관리하는 TypedDict
    """
    topic: str
    scenes: List[Dict[str, str]]
    final_video_path: str


# ============================================================================
# 재시도 데코레이터
# ============================================================================

def retry_node(max_retries: int = 2, delay: float = 1.0):
    """노드 함수에 재시도 로직을 추가하는 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(state: VideoState) -> VideoState:
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    logger.debug(f"{func.__name__} 실행 (시도 {attempt + 1}/{max_retries + 1})")
                    return func(state)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        import time
                        wait_time = delay * (attempt + 1)
                        logger.warning(
                            f"{func.__name__} 실패 (시도 {attempt + 1}/{max_retries + 1}), "
                            f"{wait_time:.1f}초 후 재시도: {e}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"{func.__name__} 최종 실패: {e}", exc_info=True)
            raise last_error
        return wrapper
    return decorator


# ============================================================================
# 캐시 데코레이터
# ============================================================================

def with_cache(stage: str, result_keys: List[str]):
    """
    노드에 캐싱 로직을 추가하는 데코레이터
    
    Args:
        stage: 캐시 단계 이름
        result_keys: 캐시할 상태 키 목록
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state: VideoState) -> VideoState:
            cache = get_cache()
            
            # 캐시 확인
            if cache:
                cached = cache.get_stage(stage)
                if cached:
                    logger.info(f"[{stage}] 캐시에서 로드 (스킵)")
                    result = {**state}
                    for key in result_keys:
                        if key in cached:
                            result[key] = cached[key]
                    return result
            
            # 실제 실행
            result = func(state)
            
            # 캐시 저장
            if cache:
                cache_data = {key: result.get(key) for key in result_keys}
                cache.set_stage(stage, cache_data)
            
            return result
        return wrapper
    return decorator


# ============================================================================
# 서브그래프 래퍼 노드
# ============================================================================

def node_script_pipeline(state: VideoState) -> VideoState:
    """
    Script Pipeline 서브그래프를 호출하는 노드
    
    이 노드는 Script Pipeline 서브그래프를 실행하여
    topic을 받아 scenes(대본 및 이미지 프롬프트)를 생성합니다.
    """
    logger.info("[메인 오케스트레이터] Script Pipeline 시작")
    script_graph = build_script_graph()
    result = script_graph.invoke(state)
    logger.info("[메인 오케스트레이터] Script Pipeline 완료")
    return result


def node_asset_pipeline(state: VideoState) -> VideoState:
    """
    Asset Generation 서브그래프를 호출하는 노드
    
    이 노드는 Asset Generation 서브그래프를 실행하여
    scenes 정보를 받아 오디오, 이미지, 모션 등 실제 에셋을 생성합니다.
    """
    logger.info("[메인 오케스트레이터] Asset Generation 시작")
    asset_graph = build_asset_graph()
    result = asset_graph.invoke(state)
    logger.info("[메인 오케스트레이터] Asset Generation 완료")
    return result


def node_video_pipeline(state: VideoState) -> VideoState:
    """
    Video Assembly 서브그래프를 호출하는 노드
    
    이 노드는 Video Assembly 서브그래프를 실행하여
    scenes 정보를 받아 최종 비디오를 생성합니다.
    """
    logger.info("[메인 오케스트레이터] Video Assembly 시작")
    video_graph = build_video_graph()
    result = video_graph.invoke(state)
    logger.info("[메인 오케스트레이터] Video Assembly 완료")
    return result


# ============================================================================
# 그래프 구성
# ============================================================================

def create_video_graph(checkpoint: bool = False) -> StateGraph:
    """
    메인 오케스트레이터 그래프를 생성하고 서브그래프들을 연결합니다.
    
    이 그래프는 3개의 서브그래프를 순차적으로 호출합니다:
    1. Script Pipeline: topic → scenes (대본 및 이미지 프롬프트)
    2. Asset Generation: scenes → scenes (오디오, 이미지, 모션 에셋 생성)
    3. Video Assembly: scenes → final_video_path (최종 비디오 생성)
    
    Args:
        checkpoint: 체크포인트 사용 여부 (메모리 기반)
        
    Returns:
        컴파일된 StateGraph
    """
    logger.info("메인 오케스트레이터 그래프 구성 중...")
    
    # 그래프 생성
    if checkpoint:
        memory = MemorySaver()
        workflow = StateGraph(VideoState).compile(checkpointer=memory)
    else:
        workflow = StateGraph(VideoState)
    
    # 서브그래프 노드 추가
    workflow.add_node("script_pipeline", node_script_pipeline)
    workflow.add_node("asset_pipeline", node_asset_pipeline)
    workflow.add_node("video_pipeline", node_video_pipeline)
    
    # 엣지 연결 (순차 실행)
    workflow.set_entry_point("script_pipeline")
    workflow.add_edge("script_pipeline", "asset_pipeline")
    workflow.add_edge("asset_pipeline", "video_pipeline")
    workflow.add_edge("video_pipeline", END)
    
    # 컴파일
    if not checkpoint:
        workflow = workflow.compile()
    
    logger.info("메인 오케스트레이터 그래프 구성 완료")
    return workflow


def create_video_graph_simple() -> StateGraph:
    """간단한 버전의 그래프 생성 (체크포인트 없음)"""
    return create_video_graph(checkpoint=False)
