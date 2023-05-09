'''
        -Quadratic Splitting Network for MR Image Reconstruction
'''
import torch.nn as nn
import torch
from einops import rearrange
import math
import warnings
from torch import einsum
import torch.nn.functional as F
from thop import profile
from timm.models.layers import DropPath, trunc_normal_, drop_path

def make_model(args, parent=False):
    a = torch.Tensor(1, 1, 256, 256)
    k = torch.Tensor(1, 1, 256, 256)
    mask = torch.Tensor(1, 1, 256, 256)

    model = GAADMMNet()
    flops, params = profile(model, (a,k,mask,))
    print('flops: ', flops/1e9, 'params: ', params/1e6)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return model

## ABLATION 
class CSE_Block(nn.Module):
    def __init__(self, in_ch, out_ch, r):
        super(CSE_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, int(in_ch / r), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(in_ch / r), in_ch, kernel_size=1),
            nn.Sigmoid()
        )
        self.tail =  nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        s = self.layer(x)
        s= s*x
        out = self.tail(s)
        return out

class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.25, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.AdaptiveAvgPool2d(1)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Conv2d(in_channels, reduce_channels, kernel_size=3, stride=1, padding=1,groups=4)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        self.spatial_select = nn.Conv2d(reduce_channels, in_channels, kernel_size=3, stride=1, padding=1,groups=4)
        self.gate_fn = gate_fn()
        self.dim =in_channels
    def forward(self, x):
        ori_x = x
        x_global = self.norm(x)
        x_global = x_global.view(-1,self.dim)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_global_4d =x_global.unsqueeze(-1).unsqueeze(-1)
        x_local = self.act_fn(self.local_reduce(x))


        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn).unsqueeze(-1).unsqueeze(-1)  # [B, 1, C]
        s_attn = self.spatial_select(x_local)
        s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn * s_attn  # [B, N, C]
        return ori_x * attn


def window_reverse(
        windows: torch.Tensor,
        original_size,
        window_size=(7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0] * window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, original_size[0] * original_size[1], C].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, -1)
    return output


def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.
    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.
    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)

class Attention(nn.Module):
    def __init__(self, dim, num_tokens=1, num_heads=8, window_size=7, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.num_tokens = num_tokens
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        # positional embedding
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward_global_aggregation(self, q, k, v):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_local(self, q, k, v, H, W):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        B, num_heads, N, C = q.shape
        ws = self.window_size
        h_group, w_group = H // ws, W // ws

        # partition to windows
        q = q.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        q = q.view(-1, num_heads, ws*ws, C)
        k = k.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        k = k.view(-1, num_heads, ws*ws, C)
        v = v.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        v = v.view(-1, num_heads, ws*ws, v.shape[-1])

        attn = (q @ k.transpose(-2, -1)) * self.scale
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws*ws, -1)

        # reverse
        x = window_reverse(x, (H, W), (ws, ws))
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward(self, x, H, W):
        B, N, C = x.shape
        NC = self.num_tokens
        # pad
        x_img, x_global = x[:, NC:], x[:, :NC]
        x_img = x_img.view(B, H, W, C)
        pad_l = pad_t = 0
        ws = self.window_size
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x_img = F.pad(x_img, (0, 0, pad_l, pad_r, pad_t, pad_b))
        Hp, Wp = x_img.shape[1], x_img.shape[2]
        x_img = x_img.view(B, -1, C)
        x = torch.cat([x_global, x_img], dim=1)

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NC:], k[:, :, NC:], v[:, :, NC:]
        q_cls, _, _ = q[:, :, :NC], k[:, :, :NC], v[:, :, :NC]

        # local window attention
        x_img = self.forward_local(q_img, k_img, v_img, Hp, Wp)
        # restore to the original size
        x_img = x_img.view(B, Hp, Wp, -1)[:, :H, :W].reshape(B, H*W, -1)
        q_img = q_img.reshape(B, self.num_heads, Hp, Wp, -1)[:, :, :H, :W].reshape(B, self.num_heads, H*W, -1)
        k_img = k_img.reshape(B, self.num_heads, Hp, Wp, -1)[:, :, :H, :W].reshape(B, self.num_heads, H*W, -1)
        v_img = v_img.reshape(B, self.num_heads, Hp, Wp, -1)[:, :, :H, :W].reshape(B, self.num_heads, H*W, -1)

        # global aggregation
        x_cls = self.forward_global_aggregation(q_cls, k_img, v_img)
        k_cls, v_cls = self.kv_global(x_cls).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        # gloal broadcast
        x_img = x_img + self.forward_global_broadcast(q_img, k_cls, v_cls)

        x = torch.cat([x_cls, x_img], dim=1)
        x = self.proj(x)
        return x



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class U_net(nn.Module):
    def __init__(self, in_ch,out_ch=96):
        super(U_net, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out

class l_nl_attn(nn.Module):
    def __init__(self, dim, num_heads, num_tokens=8, window_size=7, qkv_bias=False, drop=0., attn_drop=0.):
        super(l_nl_attn, self).__init__()

        self.attn = Attention(dim, num_heads=num_heads, num_tokens=num_tokens, window_size=window_size, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.global_token = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.dim= dim
    def forward(self, x):
        # encoder
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        global_token = self.global_token.expand(x.shape[0], -1, -1)
        x = torch.cat((global_token, x), dim=1)

        x = self.attn(x,H, W)
        x = x[:, -H*W:]
        out = x.view(-1, H, W, self.dim).permute(0, 3, 1, 2).contiguous()

        return out


class U_net_trans(nn.Module):
    def __init__(self, in_ch,out_ch, dim, num_heads, num_tokens=8, window_size=7, qkv_bias=False, drop=0., attn_drop=0.):
        super(U_net_trans, self).__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)
        self.attn1 = l_nl_attn(dim, num_heads=num_heads, num_tokens=num_tokens, window_size=window_size, qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0,groups=4)
        self.attn2 = l_nl_attn(dim, num_heads=num_heads, num_tokens=num_tokens, window_size=window_size, qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop)
        # self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()
        

    def forward(self, x):

        # encoder
        residual_1 = x
        out = self.conv1(x)
        out = self.conv2(out)
        residual_2 = out
        out =self.attn1(out)
        out = self.conv4(out)

        residual_3 = out
        out = self.conv5(out)
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(out)

        out = self.attn2(out)

        out += residual_2
        out = self.tconv4(out)
        out = self.tconv5(out)
        out += residual_1
        out = self.relu(out)
        return out


class DC_layer(nn.Module):
    def __init__(self):
        super(DC_layer, self).__init__()

    def forward(self, mask, x_rec, x_under):
        matrixones = torch.ones_like(mask.data)  # 全为1：(16, 1, 256, 256)
        x_rec_dc = (matrixones - mask) * x_rec + x_under
        out = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(x_rec_dc, dim=(-2, -1))))
        return out  # (16, 1, 256, 256)

class Resudial_B(nn.Module):
    def __init__(self, in_ch, out_ch, numB = 6):
        super(Resudial_B, self).__init__()
        self.num = numB
        # self.block =
        self.block = nn.ModuleList([])
        for _ in range(numB):   
            self.block.append(nn.Conv2d(in_ch, int(in_ch/2), kernel_size=3, padding =1))
            self.block.append(nn.ReLU())
            self.block.append(nn.Conv2d(int(in_ch/2), in_ch, kernel_size=3, padding =1))

        # for _ in range(numB):
        #     self.layer.append(self.block)
            
        self.tail = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        for i in range(self.num):
            x_temp1 = self.block[i*3](x)
            x_temp2 = self.block[i*3+1](x_temp1)
            x_temp3 = self.block[i*3+2](x_temp2)
        x=x+x_temp3
        out =self.tail(x)
        return out  # (16, 1, 256, 256)

class GAADMMNet(nn.Module):
    def __init__(self,  buffer_size=32, n_iter=18, norm='ortho',type='unet'):
        '''
        :param buffer_size: m
        :param n_iter: n
        :param n_filter: output channel for convolutions
        :param norm: 'ortho' norm for fft
        '''
        super(GAADMMNet, self).__init__()
        self.norm = norm
        self.m = buffer_size
        self.n_iter = n_iter
        # the initialization of mu may influence the final accuracy
        self.eta = nn.Parameter(0.5 * torch.ones((buffer_size, 1)))
        self.beta = nn.Parameter(0.5 * torch.ones((buffer_size, 1)))
        # self.beta2 = nn.Parameter(0.5 * torch.ones((buffer_size, 1)))
        self.DC = DC_layer()
        self.denoiser = nn.ModuleList([])
        # self.fai1 =  Resudial_B(buffer_size,buffer_size,numB=6)           
        # self.fai2 =  Resudial_B(buffer_size,buffer_size,numB=6) 
        self.fai1 =  BiAttn(buffer_size)           
        self.fai2 =  BiAttn(buffer_size)    

        self.InfoFusion1 = nn.ModuleList([])
        self.InfoFusion2 = nn.ModuleList([])
        self.InfoFusion3 = nn.ModuleList([])
        self.InfoFusion4 = nn.ModuleList([])
        for _ in range(n_iter):
            if type=='res':
                self.denoiser.append(
                    Resudial_B(buffer_size,buffer_size,numB=12),
                )
            elif type=='unet':
                # self.denoiser.append(
                #     U_net(buffer_size,buffer_size),)
                self.denoiser.append(
                    U_net_trans(buffer_size,buffer_size, dim=32,num_heads=8,num_tokens=8,window_size=7,qkv_bias=True,drop=0.,attn_drop=0., ),)
                
            
            self.InfoFusion1.append(
                CSE_Block(2*buffer_size,buffer_size,8),
            )

            self.InfoFusion2.append(
                CSE_Block(3*buffer_size,buffer_size,8),
            )

            self.InfoFusion3.append(
                CSE_Block(2*buffer_size,buffer_size,8),
            )

            self.InfoFusion4.append(
                CSE_Block(2*buffer_size,buffer_size,8),
            )


    def forward(self, img, y,mask):
        '''
        :param img: zero_filled imgs (batch, 1, h, w)
        :param y: corresponding undersampled k-space data (batch, 1, h, w)
        :param mask: sampling mask
        :return: reconstructed img
        ''' 
        
        # initialize buffer f: the concatenation of m copies of the zero-filled images
        x_0 = torch.cat([img] * self.m, 1).to(img.device)
        y_0 = torch.cat([y] * self.m, 1).to(img.device)
        
        z_klist = []
        b_klist = []
        z_klist.append(x_0)
        z_hat = x_0
        b_hat = torch.zeros_like(x_0)
        b_klist.append(b_hat)
        
        FTy = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y_0, dim=(-2, -1))))
        # n reconstruction blocks
        for i in range(self.n_iter):
            # ZB_hat = torch.fft.fftshift(torch.fft.fft2(z_hat+b_hat),dim=(-2, -1))

            x_k = self.InfoFusion1[i](torch.cat([self.fai1(z_hat+b_hat), self.fai2(FTy)], dim = 1))
            xk_temp = torch.fft.fftshift(torch.fft.fft2(x_k),dim=(-2, -1))
            x_k = self.DC(mask, xk_temp,  y)

            z_temp =  self.denoiser[i](x_k+b_hat)
            z_klist.append(z_temp)
            b_k =  self.InfoFusion2[i](torch.cat([b_hat, x_k, -z_temp],dim=1))
            b_klist.append(b_k)
            b_hat = self.InfoFusion3[i]( torch.cat([(1+self.beta[i])*b_klist[-1], -self.beta[i]*b_klist[-2] ], dim=1))
            z_hat = self.InfoFusion4[i]( torch.cat([(1+self.beta[i])*z_klist[-1], -self.beta[i]*z_klist[-2] ], dim=1))

        return x_k[:, 0:1]


