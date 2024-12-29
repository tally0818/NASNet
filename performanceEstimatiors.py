import torch
import torch.nn as nn
import numpy as np

class PerformanceEstimator():
  def __init__(self, train_loader, test_loader):
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def estimate(self, model : nn.Module) -> float:
    return 0
  
class TrainAndTest(PerformanceEstimator):
  def __init__(self, train_loader, test_loader,
                 epochs: int = 2,
                 learning_rate: float = 0.001,
                 criterion = None):
        super().__init__(train_loader, test_loader)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

  def train_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        model.train()
        total_loss = 0.0
        total_samples = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        return total_loss / total_samples

  def evaluate(self, model: nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples

        return avg_loss

  def estimate(self, model: nn.Module) -> float:
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        best_loss = np.inf

        # Training loop
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(model, optimizer)
            test_loss = self.evaluate(model)
            best_loss = min(best_loss, test_loss)

        return best_loss
  
class SynFlow(PerformanceEstimator):
  def __init__(self, train_loader, test_loader):
    super().__init__(train_loader, test_loader)

  def _set_weights_to_positive(self, model):
    for param in model.parameters():
      if param.requires_grad:
        param.data = torch.abs(param.data)

  def _setup_dummy_input(self, model):
    for batch in self.train_loader:
      x = batch [0] if isinstance(batch, (tuple, list)) else batch
      break
    dummy_input = torch.ones_like(x, device=self.device)
    return dummy_input

  def estimate(self, model: nn.Module) -> float:
    model.eval()
    model = model.to(self.device)

    original_weights = {}
    for name, param in model.named_parameters():
      if param.requires_grad:
        original_weights[name] = param.data.clone()

    self._set_weights_to_positive(model)
    dummy_input = self._setup_dummy_input(model)
    output = model(dummy_input)
    loss = output.sum()
    model.zero_grad()
    loss.backward()
    synflow_score = 0.0
    for param in model.parameters():
      if param.grad is not None:
        synflow_score += (param * param.grad).sum().item()

    for name, param in model.named_parameters():
      if param.requires_grad:
        param.data = original_weights[name]

    return -synflow_score # higher score predicts higher accuray & lower loss
  
  