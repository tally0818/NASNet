from macro_structure import *
from micro_structure import Cell
from typing import List
import numpy as np

class SearchSpace():
  def __init__(self, input_size: int, output_size: int, BackBone: List[int],
               ops: List[str], combine_methods: nn.ModuleList, B: int, init_channels: int):
    self.input_size = input_size
    self.output_size = output_size
    self.BackBone = BackBone
    self.ops = ops
    self.combine_methods = combine_methods
    self.B = B
    self.init_channels = init_channels
    self.num_operations = len(ops)
    self.num_combine_methods = len(combine_methods)

  def sample_by_action(self, action: List) -> nn.Module:

    def create_normal_cell(in1_channels, in2_channels, normal_action):
      return Cell(self.ops, self.combine_methods, self.B, normal_action, in1_channels, in2_channels)
    def create_reduction_cell(in1_channels, in2_channels, reduction_action):
      return Cell(self.ops, self.combine_methods, self.B, reduction_action, in1_channels, in2_channels, True)

    return NASNet_A(self.input_size, self.output_size, self.BackBone,
                   create_normal_cell, create_reduction_cell, action, self.init_channels)

  def _get_cell_action(self) -> List:
    num_hidden_states = 2  
    cell_action = []
    for b in range(self.B):
      input1 = np.random.randint(num_hidden_states)
      input2 = np.random.randint(num_hidden_states)
      op1 = np.random.randint(self.num_operations)
      op2 = np.random.randint(self.num_operations)
      combine = np.random.randint(self.num_combine_methods)
      cell_action.extend([input1, input2, op1, op2, combine])
      num_hidden_states += 1  
    return cell_action

  def sample_action(self):
    normal_action = self._get_cell_action()
    reduction_action = self._get_cell_action()
    return normal_action + reduction_action 

  def sample(self) -> nn.Module:
    action = self.sample_action()
    return self.sample_by_action(action)