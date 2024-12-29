from operations import *

class Block(nn.Module):
  def __init__(self, op1: nn.Module, op2: nn.Module, combine: nn.Module):
    super(Block, self).__init__()
    self.op1 = op1
    self.op2 = op2
    self.combine = combine
    self.out_channels = self._calculate_out_channels()


  def _calculate_out_channels(self):
    '''
    calculate out channels
    '''
    if isinstance(self.combine, Concat):
      return self.op1.C_out + self.op2.C_out
    return max(self.op1.C_out, self.op2.C_out)

  def _match_shapes(self, x1, x2):
    '''
    combine 가능하게 두 출력의 차원을 맞춤
    '''
    _, c1, h1, w1 = x1.size()
    _, c2, h2, w2 = x2.size()
    device = x1.device

    # 공간 차원 맞추기
    target_size = min(h1, h2)
    if h1 != target_size:
      stride = h1 // target_size
      x1 = nn.Conv2d(c1, c1, 1, stride=stride).to(device)(x1)
    if h2 != target_size:
      stride = h2 // target_size
      x2 = nn.Conv2d(c2, c2, 1, stride=stride).to(device)(x2)

    # 채널 차원 맞추기 (Add의 경우)
    if isinstance(self.combine, Add):
      target_channels = max(c1, c2)
      if c1 != target_channels:
        x1 = nn.Conv2d(c1, target_channels, 1).to(device)(x1)
      if c2 != target_channels:
        x2 = nn.Conv2d(c2, target_channels, 1).to(device)(x2)

    return x1, x2

  def check(self, out1, out2):
    '''
    C_out 으로 지정된 채널수 만큼이 나오는지,
    나오지 않는다면 1x1_conv로 수정
    '''
    if out1.size()[1] != self.op1.C_out:
      out1 = nn.Conv2d(out1.size()[1], self.op1.C_out, 1, stride=1).to(out1.device)(out1)
    if out2.size()[1] != self.op2.C_out:
      out2 = nn.Conv2d(out2.size()[1], self.op2.C_out, 1, stride=1).to(out2.device)(out2)
    return out1, out2

  def forward(self, x1, x2):
    out1 = self.op1(x1)
    out2 = self.op2(x2)
    out1, out2 = self.check(out1, out2)
    out1, out2 = self._match_shapes(out1, out2)
    out = self.combine(out1, out2)
    return out