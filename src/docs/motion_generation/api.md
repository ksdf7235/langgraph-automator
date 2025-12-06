# Motion Generation API 명세

## Public 클래스

### `MotionGenerator`

Wan2.2 I2V를 사용한 모션 생성 클래스

**주요 메서드:**

- `generate_motion_frames()` - 프레임 시퀀스 생성
- `generate_all_motions()` - 모든 장면의 모션 생성

---

## Public 함수

### `node_motion_generator(state: Dict) -> Dict`

LangGraph 노드 함수: 모션 생성
