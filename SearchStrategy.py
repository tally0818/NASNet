import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class RNNController(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_blocks: int,  # B
        max_hidden_states: int,  # 이전 블록들의 hidden state 선택 시 최대 개수
        num_operations: int,  # 가능한 연산의 수
        num_combine_methods: int,  # 가능한 결합 방법의 수
        temperature: float = 1.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.max_hidden_states = max_hidden_states
        self.num_operations = num_operations
        self.num_combine_methods = num_combine_methods
        self.temperature = temperature

        # 각 블록당 5개의 결정을 하고, Normal Cell과 Reduction Cell 2개를 생성
        self.sequence_length = 2 * num_blocks * 5

        # Controller LSTM
        self.lstm = nn.LSTMCell(
            input_size=hidden_size,
            hidden_size=hidden_size
        )

        # 각 결정을 위한 출력 레이어들
        self.hidden_state_selector = nn.Linear(hidden_size, max_hidden_states)
        self.operation_selector = nn.Linear(hidden_size, num_operations)
        self.combine_selector = nn.Linear(hidden_size, num_combine_methods)

        # 시작 토큰 (learnable)
        self.start_token = nn.Parameter(torch.randn(1, hidden_size))

    def sample_logits(self, logits: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """주어진 logits에서 샘플링을 수행"""
        probs = F.softmax(logits / self.temperature, dim=-1)

        if self.training:
            distribution = torch.distributions.Categorical(probs)
            sample = distribution.sample()
            log_prob = distribution.log_prob(sample)
            return sample, log_prob
        else:
            sample = torch.argmax(probs, dim=-1)
            return sample, None

    def get_selector(self, step: int, num_possible_inputs: int) -> nn.Module:
        """현재 단계에 맞는 selector를 반환"""
        step = step % 5  # 각 블록의 5가지 단계 중 어느 것인지
        if step < 2:  # hidden state 선택 단계
            return lambda x: self.hidden_state_selector(x)[..., :num_possible_inputs]
        elif step < 4:  # operation 선택 단계
            return self.operation_selector
        else:  # combine method 선택 단계
            return self.combine_selector

    def forward(self, batch_size: int = 1) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        아키텍처를 생성하는 forward pass

        Returns:
            actions: (batch_size, sequence_length) 크기의 텐서
            log_probs: 학습 모드일 때만 반환되는 log probability 텐서
        """
        device = next(self.parameters()).device

        # LSTM 상태 초기화
        h_t = torch.zeros(batch_size, self.hidden_size).to(device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(device)

        # 입력 초기화 (start token)
        x_t = self.start_token.expand(batch_size, -1)

        actions = []
        log_probs = [] if self.training else None

        # Normal Cell과 Reduction Cell 각각에 대해
        for cell in range(2):
            num_prev_states = 2  # 시작시 hi, hi-1 두 개의 상태

            # 각 블록에 대해
            for block in range(self.num_blocks):
                # 블록당 5가지 결정
                for step in range(5):
                    # LSTM 스텝
                    h_t, c_t = self.lstm(x_t, (h_t, c_t))

                    # 현재 단계에 맞는 selector 가져오기
                    selector = self.get_selector(step, num_prev_states)

                    # Logits 계산 및 샘플링
                    logits = selector(h_t)
                    action, log_prob = self.sample_logits(logits)

                    actions.append(action)
                    if self.training:
                        log_probs.append(log_prob)

                    # 다음 스텝의 입력으로 현재 hidden state 사용
                    x_t = h_t

                # 블록이 끝날 때마다 가능한 hidden state 수 증가
                num_prev_states += 1

        actions = torch.stack(actions, dim=1)
        if self.training:
            log_probs = torch.stack(log_probs, dim=1)

        return actions, log_probs

    def decode_actions(self, actions: torch.Tensor) -> dict:
        """
        컨트롤러가 생성한 액션을 SearchSpace에서 사용할 수 있는 형식으로 디코딩

        Args:
            actions: (sequence_length,) 크기의 텐서

        Returns:
            dict: Normal Cell과 Reduction Cell의 구조를 담은 딕셔너리
        """
        actions = actions.cpu().numpy()
        cells = {}

        # Normal Cell과 Reduction Cell 각각에 대해
        for cell_type in ['normal', 'reduction']:
            cell_actions = actions[len(actions)//2:] if cell_type == 'reduction' else actions[:len(actions)//2]
            cell_struct = []

            # 각 블록의 구조 디코딩
            for block in range(self.num_blocks):
                start_idx = block * 5
                block_struct = {
                    'input1': cell_actions[start_idx],
                    'input2': cell_actions[start_idx + 1],
                    'op1': cell_actions[start_idx + 2],
                    'op2': cell_actions[start_idx + 3],
                    'combine': cell_actions[start_idx + 4]
                }
                cell_struct.append(block_struct)

            cells[f'{cell_type}_cell'] = cell_struct

        return cells