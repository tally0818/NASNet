from performanceEstimatiors import *
from RNNController import Controller
from SearchSapce import SearchSpace
import queue
from collections import deque
from typing import List

class Random_Search():
  def __init__(self, search_space : SearchSpace, performance_estimator : PerformanceEstimator):
    self.search_space = search_space
    self.performance_estimator = performance_estimator

  def search(self, num_searches : int) -> nn.Module:
    best_model = None
    best_loss = np.inf
    for _ in range(num_searches):
      model = self.search_space.sample()
      loss = self.performance_estimator.estimate(model)
      if loss < best_loss:
        best_loss = loss
        best_model = model
    return best_model

class Regularized_Evolution():
  def __init__(self, search_space : SearchSpace, pop_size : int, performance_estimator : PerformanceEstimator):
    self.search_space = search_space
    self.pop_size = pop_size
    self.performance_estimator = performance_estimator
    self.population = queue.Queue()
    self.history = []
    while self.population.qsize() < self.pop_size:
      model_arch = self.search_space.sample_action()
      model = self.search_space.sample_by_action(model_arch)
      self.population.put(model)
      loss = self.performance_estimator.estimate(model)
      self.history.append([model_arch, loss])

  def _get_best_in_sample(self, sample):
    best_arch = sample[0][0]
    best_loss = sample[0][1]
    for i in range(len(sample)):
      if best_loss > sample[i][1]:
        best_arch = sample[i][0]
        best_loss = sample[i][1]
    return best_arch

  def _op_mutate(self, arch : List) -> List:
    block_idx = np.random.randint(self.search_space.B)
    op_idx = np.random.randint(2) + 2
    target_op_idx = 5 * block_idx + op_idx  # action 은 (input1, input2, op1, op2, combine) 단위 !
    arch[target_op_idx] = np.random.randint(self.search_space.num_operations)
    return arch

  def _hs_mutate(self, arch : List) -> List:
    block_idx = np.random.randint(self.search_space.B)
    input_idx = np.random.randint(2)
    target_input_idx = 5 * block_idx + input_idx  # action 은 (input1, input2, op1, op2, combine) 단위 !
    arch[target_input_idx] = np.random.randint(block_idx + 2)
    return arch

  def _mutate(self, arch : List) -> List:
    if np.random.rand() < 0.5:
      return self._op_mutate(arch)
    return self._hs_mutate(arch)

  def _discard(self, dead):
    history = []
    for i in range(len(self.history)):
      if self.history[i][0] != dead:
        history.append(self.history[i])
    self.history = history

  def search(self, num_generations : int, sample_size : int) -> nn.Module:
    for generation in range(num_generations):
      sample = []
      while len(sample) < sample_size:
        idx = np.random.randint(self.pop_size)
        sample.append(self.history[idx])
      parent_arch = self._get_best_in_sample(sample)
      child_arch = self._mutate(parent_arch)
      child_model = self.search_space.sample_by_action(child_arch)
      child_loss = self.performance_estimator.estimate(child_model)
      self.history.append([child_arch, child_loss])
      self.population.put(child_model)
      dead = self.population.get()
    best_arch = self._get_best_in_sample(self.history)
    best_model = self.search_space.sample_by_action(best_arch)
    return best_model


class SimplePPOTrainer():
  def __init__(self,
               search_space : SearchSpace,
               controller: Controller,
               performance_estimator: PerformanceEstimator,
               learning_rate: float = 0.0003,
               clip_epsilon: float = 0.2,
               entropy_coef: float = 0.01,
               batch_size: int = 32,
               n_epochs: int = 4,
               ):
    self.search_space = search_space
    self.controller = controller
    self.performance_estimator = performance_estimator
    self.optimizer = optim.Adam(controller.parameters(), lr=learning_rate)
    self.clip_epsilon = clip_epsilon
    self.entropy_coef = entropy_coef
    self.batch_size = batch_size
    self.n_epochs = n_epochs

  def calculate_entropy(self, logits: torch.Tensor) -> torch.Tensor:
    """주어진 logits에 대한 엔트로피 계산"""
    probs = F.softmax(logits / self.controller.temperature, dim=-1)
    dist = torch.distributions.Categorical(probs)
    return dist.entropy().mean()

  def train_step(self):
    self.controller.train()
    with torch.no_grad():
      actions, old_log_probs = self.controller.forward()
    actions = actions.squeeze()

    model = self.search_space.sample_by_action(actions)
    reward = self.performance_estimator.estimate(model)


    rewards = torch.tensor([reward] * self.controller.sequence_length, device=actions.device)

    # PPO update
    for _ in range(self.n_epochs):
      _, new_log_probs = self.controller(batch_size=1)
      ratio = (new_log_probs - old_log_probs).exp()

      # PPO clipped objective
      surr1 = ratio * rewards
      surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * rewards
      policy_loss = torch.min(surr1, surr2).mean()  # reward is a loss, not a advantage func

      # 엔트로피 계산
      total_entropy = 0.0
      num_decisions = 0

      for step in range(self.controller.sequence_length):
        step_in_block = step % 5
        num_prev_states = 2 + (step // 5)

        if step_in_block < 2:
          logits = self.controller.hidden_state_selector(torch.zeros(1, self.controller.hidden_size).to(actions.device))
          logits = logits[..., :num_prev_states]
        elif step_in_block < 4:
          logits = self.controller.operation_selector(torch.zeros(1, self.controller.hidden_size).to(actions.device))
        else:
          logits = self.controller.combine_selector(torch.zeros(1, self.controller.hidden_size).to(actions.device))

        total_entropy += self.calculate_entropy(logits)
        num_decisions += 1


      entropy = total_entropy / num_decisions

      # 전체 loss 계산 (policy loss - entropy_coef * entropy)
      loss = policy_loss - self.entropy_coef * entropy # gradient descent

      # Optimization step
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    return reward

  def train(self, n_steps: int, verbose: bool = True):
    rewards = []
    moving_avg_reward = deque(maxlen=100)

    for step in range(n_steps):
      reward = self.train_step()
      rewards.append(reward)
      moving_avg_reward.append(reward)

      if verbose and (step + 1) % 10 == 0:
        avg_reward = sum(moving_avg_reward) / len(moving_avg_reward)
        print(f"Step {step+1}/{n_steps} | " f"Average Reward: {avg_reward:.4f}")

    return rewards

  def sample(self, num_epoch: int) -> nn.Module:
    self.train(num_epoch)
    self.controller.eval()
    with torch.no_grad():
      actions = self.controller.forward(batch_size=1)[0][0]
    model = self.search_space.sample_by_action(actions)
    return model
