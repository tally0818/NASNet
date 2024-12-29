import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Controller(nn.Module):
  def __init__(self,
               hidden_size: int,
               num_blocks: int,  # B
               num_operations: int,  # 가능한 연산의 수
               num_combine_methods: int,  # 가능한 결합 방법의 수
               temperature: float = 1.0
               ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_blocks = num_blocks
    self.num_operations = num_operations
    self.num_combine_methods = num_combine_methods
    self.temperature = temperature

    self.sequence_length = 2 * num_blocks * 5

    self.lstm = nn.LSTMCell(input_size=hidden_size,
                            hidden_size=hidden_size
                            )
    self.hidden_state_selector = nn.Linear(hidden_size, self.num_blocks + 2)
    self.operation_selector = nn.Linear(hidden_size, num_operations)
    self.combine_selector = nn.Linear(hidden_size, num_combine_methods)

    self.start_token = nn.Parameter(torch.randn(1, hidden_size))

  def sample_logits(self, logits: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
    step = step % 5
    if step < 2:
      return lambda x: self.hidden_state_selector(x)[..., :num_possible_inputs]
    elif step < 4:
      return self.operation_selector
    else:
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

    x_t = self.start_token.expand(batch_size, -1)

    actions = []
    log_probs = [] if self.training else None


    for cell in range(2):
      num_prev_states = 2


      for block in range(self.num_blocks):
        for step in range(5):
          h_t, c_t = self.lstm(x_t, (h_t, c_t))
          selector = self.get_selector(step, num_prev_states)
          logits = selector(h_t)
          action, log_prob = self.sample_logits(logits)
          actions.append(action)
          if self.training:
            log_probs.append(log_prob)
          x_t = h_t
        num_prev_states += 1

    actions = torch.stack(actions, dim=1)
    if self.training:
      log_probs = torch.stack(log_probs, dim=1)

    return actions, log_probs