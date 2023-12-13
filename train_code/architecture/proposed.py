import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.conv11 = nn.Conv1d(1, 1, 3, 1, 1, bias=False)
        self.conv12 = nn.Conv1d(1, 1, 3, 1, 1, bias=False)
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim_head * heads, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x):
        b, h, w, c = x.shape
        x = x.reshape(b * h * w, 1, c)
        # x = self.conv11(self.conv12(x))
        x = x.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                      (q_inp, k_inp, v_inp))
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, self.num_heads * self.dim_head).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, f=3, bias=False):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*f)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x.permute(0, 3, 1, 2))
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x.permute(0, 2, 3, 1)

class TransformerBlock(nn.Module):
    def __init__(self, dim=32, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim=dim, dim_head=dim//3, heads=3)),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for (att1, ff) in self.blocks:
            x = att1(x) + x
            x = ff(x) + x
        x = x.permute(0, 3, 1, 2)
        return x

class DeepFeatureExtraction(nn.Module):
    def __init__(self, ksize=3, dim=31, n=1):
        super().__init__()

        self.conv11 = nn.Conv2d(dim, dim//2, ksize, 1, ksize//2, bias=False)
        self.conv12 = nn.Conv2d(dim//2, 1, ksize, 1, ksize//2, bias=False)
        self.conv21 = nn.Conv2d(dim, dim//2, 4, 2, 1, bias=False)
        self.conv22 = nn.ConvTranspose2d(dim//2, 1, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.conv31 = nn.Conv2d(dim, dim//2, 8, 4, 2, bias=False)
        self.conv32 = nn.ConvTranspose2d(dim//2, 1, stride=4, kernel_size=4, padding=0, output_padding=0)
        self.conv0 = nn.Conv2d(3, 1, 1, 1, 0, bias=False)
        self.relu = GELU()
        self.conv = nn.Conv2d(dim, dim, ksize, 1, ksize//2, bias=True)
        self.tb = TransformerBlock(dim=dim, num_blocks=n)

    def forward(self, x):
        x1 = self.relu(self.conv11(x))
        x1 = self.relu(self.conv12(x1))
        x0 = x1
        x1 = self.relu(self.conv21(x))
        x1 = self.relu(self.conv22(x1))
        x0 = torch.cat([x0, x1], dim=1)
        x1 = self.relu(self.conv31(x))
        x1 = self.relu(self.conv32(x1))
        x0 = torch.cat([x0, x1], dim=1)
        x0 = torch.sigmoid(self.conv0(x0))
        x0 = x * x0
        x0 = self.tb(self.conv(x0) + x)
        return x0

class Main(nn.Module):
    def __init__(self):
        super().__init__()
        dim0 = 66
        self.k = 1
        self.conv11 = nn.Conv2d(3, dim0, 1, 1, 0, bias=False)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.conv0 = nn.Conv2d(3, 31, 3, 1, 1, bias=False)
        self.conv1 = nn.Conv2d(31, dim0, 3, 1, 1, bias=False)
        self.df1 = DeepFeatureExtraction(ksize=3, dim=dim0, n=1)
        self.df2 = DeepFeatureExtraction(ksize=3, dim=dim0, n=1)
        self.df3 = DeepFeatureExtraction(ksize=3, dim=dim0, n=1)
        self.df4 = DeepFeatureExtraction(ksize=3, dim=dim0, n=1)
        self.con1 = nn.Conv2d(dim0*2, dim0, 3, 1, 1, bias=False)
        self.df5 = DeepFeatureExtraction(ksize=3, dim=dim0, n=1)
        self.con2 = nn.Conv2d(dim0*2, dim0, 3, 1, 1, bias=False)
        self.df6 = DeepFeatureExtraction(ksize=3, dim=dim0, n=1)
        self.con3 = nn.Conv2d(dim0*2, dim0, 3, 1, 1, bias=False)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(dim0, 31, 3, 1, 1, bias=False)

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        y = torch.ones(x.shape).cuda()
        y = x - y * self.avg(x)
        y = self.conv11(y)

        x = self.conv1(self.conv0(x))
        # x= torch.cat([self.tb(x, x, x), x], dim=1)
        x1 = self.df1(x + self.k * y)
        x2 = self.df2(x1 + self.k * y)
        x3 = self.df6(self.df3(x2 + self.k * y))
        x3 = torch.cat([x2, x3], dim=1)
        x4 = self.df4(self.con1(x3))
        x4 = torch.cat([x1, x4], dim=1)
        x5 = self.df5(self.con2(x4))
        x5 = torch.cat([x, x5], dim=1)
        out = self.conv2(self.relu(self.con3(x5)))
        return out[:, :, :h_inp, :w_inp]
