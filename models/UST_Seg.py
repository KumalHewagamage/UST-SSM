import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from point_4d_convolution import *

import math
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from functools import partial
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .block import Block
from .Curve import *
from ipdb import set_trace as st
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class UST(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                               
                 temporal_kernel_size, temporal_stride,                                 
                 dim, depth, heads,                               
                 mlp_dim, num_classes, dropout, hos_branches_num, encoder_channel):                         
        super().__init__()
        self.hos_branches_num = hos_branches_num
        self.depth = depth

        self.inner = 32
        self.tokens = 64
        self.hidden = 1


        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim//2], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride//8,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=1, temporal_padding=[0, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')
        self.tube_embedding1 = P4DConv(in_planes=dim//2, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride//4,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embed = nn.Sequential(
            nn.Linear(4, 128),
            nn.SiLU(),
            nn.Linear(128, dim + self.hidden)
        )



        self.ssm_blocks = nn.ModuleList()
        initial_k_group_size = 12
        for i in range(self.hos_branches_num):
            branch_ssm_blocks = nn.ModuleList()
            for _ in range(self.depth):
                k_group_size = max(1, initial_k_group_size // (2 ** i))
                ssm_block = AggregationSSM(
                    dim=dim + self.hidden,
                    num_group=768,
                    num_heads=heads,
                    drop_path=0.1,
                    k_group_size=k_group_size,
                )
                branch_ssm_blocks.append(ssm_block)
            self.ssm_blocks.append(branch_ssm_blocks)



        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.hos_branches_num * (dim+self.hidden) ),
            nn.Linear(self.hos_branches_num * (dim+self.hidden), mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )



        self.route = nn.Sequential(
         PointNet(dim,self.tokens),
         nn.LogSoftmax(dim=-1)
        )


        self.embeddingB = nn.Embedding(self.tokens, self.inner) 
        self.embeddingB.weight.data.uniform_(-1 / self.tokens, 1 / self.tokens)

        self.token1 = nn.Embedding(self.inner, self.hidden)
        self.token1.weight.data.uniform_(-1 / self.inner, 1 / self.inner)


        # Segmentation decoder with proper channel dimensions
        # Encoder outputs: dim + hidden = 161
        self.deconv3 = P4DTransConv(in_planes=dim + self.hidden,
                                    mlp_planes=[256, 256],
                                    mlp_batch_norm=[True, True, True],
                                    mlp_activation=[True, True, True],
                                    original_planes=dim + self.hidden)

        self.deconv2 = P4DTransConv(in_planes=256,
                                    mlp_planes=[128, 128],
                                    mlp_batch_norm=[True, True, True],
                                    mlp_activation=[True, True, True],
                                    original_planes=dim)

        self.deconv1 = P4DTransConv(in_planes=128,
                                    mlp_planes=[64, 64],
                                    mlp_batch_norm=[True, True, True],
                                    mlp_activation=[True, True, True],
                                    original_planes=dim//2)

        # Point-wise segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, num_classes, 1)
        )

    def forward(self, input):
        device = input.get_device()
        
        # Store intermediate encoder features for skip connections
        encoder_features = []
        encoder_xyzs = []
        
        # First conv layer
        xyzs0, features0 = self.tube_embedding(input)
        encoder_xyzs.append(xyzs0)
        encoder_features.append(features0)
        
        # Second conv layer
        xyzs1, features1 = self.tube_embedding1(xyzs0, features0) 
        encoder_xyzs.append(xyzs1)
        encoder_features.append(features1)
        
        xyzs, features = xyzs1, features1
        B, L, n, _ = xyzs.shape
        C = features.shape[2]

    
        features = features.permute(0, 1, 3, 2)
        features = features.reshape(B, L * n, -1)
        xyzs = xyzs.reshape(B, L * n, -1)
      
        pred_route = self.route(features)  

        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  
        full_embedding1 = self.embeddingB.weight @ self.token1.weight  


        prompt = torch.matmul(cls_policy, full_embedding1).view(B, L * n, self.hidden)    

        features = torch.cat([features, prompt], dim=2)


        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False) 
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)

        sorted_indices_expanded = x_sort_indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]) 
        features = torch.gather(features, dim=1, index=sorted_indices_expanded)  

        if xyzs is not None:
            sorted_indices_expanded_xyz = x_sort_indices.unsqueeze(-1).expand(-1, -1, xyzs.shape[-1])
            xyzs = torch.gather(xyzs, dim=1, index=sorted_indices_expanded_xyz)  



        features = features.reshape(B, L, -1, n)
        features = features.permute(0, 1, 3, 2)
        xyzs = xyzs.reshape(B, L, n, -1)
 
        xyzts = []
        xyzs_split = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs_split = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs_split]
        for t, xyz in enumerate(xyzs_split):
            t_val = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t + 1)
            t_val = torch.div(t_val, xyzs.shape[1], rounding_mode='floor')
            xyzt = torch.cat(tensors=(xyz, t_val), dim=2)
            xyzts.append(xyzt)

        xyzts = torch.stack(tensors=xyzts, dim=1)
        pos = self.pos_embed(xyzts)

        features = features.permute(0, 1, 3, 2)

        downsampling_factors = [2 ** i for i in range(self.hos_branches_num)]
        outputs = []
        main_x_output = None  

        for i, downsampling_factor in enumerate(downsampling_factors):
            if downsampling_factor == 1:
                indices = list(range(L))
            else:
                indices = [max(0, min(L - 1, k)) for k in range(0, L, downsampling_factor)]

            xyzs_branch = xyzs[:, indices, :, :]  # [B, W, n, 3]
            features_branch = features[:, indices, :, :]  # [B, W, C, n]


            W = xyzs_branch.shape[1]

            xyzs_branch = xyzs_branch.reshape(B, W * n, 3)
            
            pos_branch = pos[:, indices, :, :]
            pos_branch = pos_branch.reshape(B, W * n, pos_branch.shape[-1])


            features_branch = features_branch.reshape(B, W * n, C+self.hidden)
            x_branch = features_branch + pos_branch  # [B, W*n, C]

            x_output = x_branch
    
            for ssm_block in self.ssm_blocks[i]:
                x_output = ssm_block(center=xyzs_branch, x=x_output)  # [B, W*n, C_out]

            if downsampling_factor == 1:
                main_x_output = x_output  # [B, L*n, C_out]

            output_branch = torch.max(x_output, dim=1, keepdim=False)[0]  # [B, C_out]
            outputs.append(output_branch)


        feat = main_x_output.reshape(B, L, n, -1)  # [B, L, n, C_out]

        outputs_concat = outputs  # [key_output, output0, output1, output2, ...]
        output_token = torch.cat(outputs_concat, dim=1)  # [B, dim//2 + hos_branches_num * C_out]

        # === Segmentation Decoder ===
        # Start from encoder output: feat [B, L, n, dim+hidden]
        # Reshape for decoder input
        feat_decoder = feat.reshape(B, L * n, -1)  # [B, L*n, dim+hidden]
        xyzs_decoder = xyzs.reshape(B, L * n, 3)   # [B, L*n, 3]
        
        # Decoder stage 3: upsample from encoder output to encoder layer 1
        xyzs_d3 = xyzs_decoder.reshape(B, L, n, 3)
        feat_d3 = feat_decoder.reshape(B, L, n, -1).permute(0, 2, 1, 3).reshape(B, n, L, -1)
        
        xyzs_skip2 = encoder_xyzs[1]  # [B, L, n, 3] from tube_embedding1
        feat_skip2 = encoder_features[1]  # [B, L, C, n]
        
        xyzs_d2, feat_d2 = self.deconv3(xyzs_d3, xyzs_skip2, feat_d3, feat_skip2)
        
        # Decoder stage 2: upsample to encoder layer 0
        xyzs_skip1 = encoder_xyzs[0]  # [B, L, n, 3] from tube_embedding
        feat_skip1 = encoder_features[0]  # [B, L, C, n]
        
        xyzs_d1, feat_d1 = self.deconv2(xyzs_d2, xyzs_skip1, feat_d2, feat_skip1)
        
        # Decoder stage 1: upsample to original input resolution
        input_xyzs = input[:, :, :, :3]  # [B, L, N, 3] original input points
        input_features = torch.zeros(B, L, self.dim//2, input.shape[2], device=device)  # dummy features
        
        xyzs_d0, feat_d0 = self.deconv1(xyzs_d1, input_xyzs, feat_d1, input_features)
        
        # Point-wise segmentation: [B, L, C, N] -> [B, L, N, num_classes]
        B_d, L_d, C_d, N_d = feat_d0.shape
        feat_flat = feat_d0.reshape(B_d * L_d, C_d, N_d)  # [B*L, C, N]
        seg_logits = self.seg_head(feat_flat)  # [B*L, num_classes, N]
        seg_logits = seg_logits.permute(0, 2, 1)  # [B*L, N, num_classes]
        seg_logits = seg_logits.reshape(B_d, L_d, N_d, -1)  # [B, L, N, num_classes]

        return output_token, seg_logits





class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)  
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        batch_size = x.size(0)
        k = x.size(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = torch.max(x, 2)[0]  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = x.view(batch_size, k, k)  
        return x

class PointNet(nn.Module):
    def __init__(self, input_channels, tokens, num_classes=None):
        super(PointNet, self).__init__()
        
        self.tnet = TNet(k=input_channels)

        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, tokens)  

        self.tokens = tokens
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):

        x = x.permute(0, 2,1)
        trans = self.tnet(x)
        x = torch.bmm(x.transpose(2, 1), trans)  
        x = x.permute(0, 2,1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.permute(0, 2,1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  
        
        return x
