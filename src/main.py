"""
YouTube Shorts 자동 생성 에이전트 - 메인 엔트리 포인트
입력 → 그래프 실행 → 출력만 담당합니다.
"""

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import Config
from src.utils.logger import setup_logger
from src.utils.cache import init_cache
from src.workflow.graph import create_video_graph_simple, VideoState

# 로거 설정
logger = setup_logger()


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="YouTube Shorts 자동 생성 에이전트")
    parser.add_argument("--topic", "-t", type=str, default="AI의 미래", help="비디오 주제")
    parser.add_argument("--no-cache", action="store_true", help="캐시 사용 안 함 (처음부터 새로 생성)")
    parser.add_argument("--clear-cache", action="store_true", help="캐시 초기화 후 실행")
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("YouTube Shorts 자동 생성 에이전트")
    logger.info("=" * 60)
    
    try:
        # 설정 검증
        Config.validate()
        logger.info("설정 검증 완료")
        
        # 캐시 초기화
        cache = init_cache(args.topic)
        
        if args.clear_cache:
            cache.clear()
            logger.info("캐시가 초기화되었습니다.")
        elif args.no_cache:
            cache.clear()
            logger.info("캐시 사용 안 함 모드")
        else:
            # 캐시 상태 출력
            status = cache.get_status()
            cached_stages = [k for k, v in status.items() if v]
            if cached_stages:
                logger.info(f"캐시된 단계: {', '.join(cached_stages)}")
        
        # 초기 상태 설정
        initial_state: VideoState = {
            "topic": args.topic,
            "scenes": [],
            "final_video_path": ""
        }
        
        # 그래프 생성
        logger.info("워크플로우 그래프 생성 중...")
        graph = create_video_graph_simple()
        
        # 그래프 실행
        logger.info(f"주제: {initial_state['topic']}")
        logger.info("워크플로우 시작...\n")
        
        final_state = graph.invoke(initial_state)
        
        # 결과 출력
        logger.info("\n" + "=" * 60)
        logger.info("워크플로우 완료!")
        logger.info("=" * 60)
        logger.info(f"최종 비디오 경로: {final_state['final_video_path']}")
        logger.info(f"생성된 장면 수: {len(final_state['scenes'])}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n사용자에 의해 중단되었습니다.")
        return 130
        
    except Exception as e:
        logger.error(f"\n오류 발생: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
