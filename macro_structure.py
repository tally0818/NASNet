from micro_structure import *

class NASNet_A(nn.Module):
  def __init__(self, input_size, output_size, BackBone, Normal_cell, Reduction_cell, action, init_channels):
    '''
    action <- 길이 10B
    '''
    super(NASNet_A, self).__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.backbone = BackBone
    self.normal_cell = Normal_cell
    self.reduction_cell = Reduction_cell
    self.channels = [init_channels, init_channels]
    self.action = action
    self.stem = nn.Sequential(
        nn.Conv2d(input_size, init_channels, 1, padding=1),
        nn.BatchNorm2d(init_channels)
    )

    self.cells = nn.ModuleList()

    for cell_idx, num_cells in enumerate(self.backbone):
      for _ in range(num_cells):
        if cell_idx % 2 == 0:
          cell = Normal_cell(self.channels[-2], self.channels[-1], self.action[:len(self.action)//2])
        else:
          cell = Reduction_cell(self.channels[-2], self.channels[-1], self.action[len(self.action)//2:])
        self.cells.append(cell)
        self.channels.append(cell.out_channels)

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Sequential(
            nn.Linear(self.channels[-1], 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, output_size)
        )

  def forward(self, x):
    x = self.stem(x)
    prev_prev = prev = x

    for cell in self.cells:
      out = cell(prev_prev, prev)
      prev_prev, prev = prev, out
    out = self.global_pooling(prev)
    out = out.view(out.size(0), -1)
    out = self.classifier(out)
    return F.softmax(out, dim=1)