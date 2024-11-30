from SearchStrategy import RNNController


# Controller 초기화
controller = RNNController(
    hidden_size=100,
    num_blocks=5,          # B=5
    max_hidden_states=7,   # 최대 7개의 이전 hidden states 선택 가능
    num_operations=7,      # 7가지 가능한 연산
    num_combine_methods=2  # 2가지 결합 방법
)

# 아키텍처 샘플링
actions, _ = controller.forward(batch_size=1)
actions = actions.squeeze(0)  # 배치 차원 제거

# 액션을 SearchSpace에서 사용할 수 있는 형식으로 디코딩
arch_spec = controller.decode_actions(actions)

print(arch_spec)