import torch.nn as nn

class Generator(nn.Module):
  def __init__(self, z_size, g_chnl, out_chnl):
      super(Generator, self).__init__()
      self.model = nn.Sequential(
        # (B, z_size, 1, 1)
        nn.ConvTranspose2d(z_size, g_chnl*8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(g_chnl*8),
        nn.ReLU(inplace=True),

        # (B, 8*g_chnl, 4, 4)
        nn.ConvTranspose2d(g_chnl*8, g_chnl*4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(g_chnl*4),
        nn.ReLU(inplace=True),

        # (B, 4*g_chnl, 8, 8)
        nn.ConvTranspose2d(g_chnl*4, g_chnl*2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(g_chnl*2),
        nn.ReLU(inplace=True),

        # (B, 2*g_chnl, 16, 16)
        nn.ConvTranspose2d(g_chnl*2, g_chnl*1, 4, 2, 1, bias=False),
        nn.BatchNorm2d(g_chnl*1),
        nn.ReLU(inplace=True),

        # (B, 1*g_chnl, 32, 32)
        nn.ConvTranspose2d(g_chnl*1, out_chnl, 4, 2, 1, bias=False),

        # (B, out_chnl, 64, 64)
        nn.Tanh()
      )
  def forward(self, x):
    img = self.model(x)
    return img

class Discriminator(nn.Module):
  def __init__(self, d_chnl, in_chnl):
      super(Discriminator, self).__init__()
      self.model = nn.Sequential(
        # (B, 3, 64, 64)
        nn.Conv2d(in_chnl, d_chnl, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        
        # (B, d_chnl, 32, 32)
        nn.Conv2d(d_chnl, d_chnl*2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(d_chnl*2),
        nn.LeakyReLU(0.2, inplace=True),
        
        # (B, 2*d_chnl, 16, 16)
        nn.Conv2d(d_chnl*2, d_chnl*4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(d_chnl*4),
        nn.LeakyReLU(0.2, inplace=True),
        
        # (B, 4*d_chnl, 8, 8)
        nn.Conv2d(d_chnl*4, d_chnl*8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(d_chnl*8),
        nn.LeakyReLU(0.2, inplace=True),
        
        # (B, 8*d_chnl, 4, 4)
        nn.Conv2d(d_chnl*8, 1, 4, 1, 0, bias=False),

        # (B, 1, 1, 1)
        nn.Sigmoid()
      )
  def forward(self, x):
    output = self.model(x)
    return output.view(-1,1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)