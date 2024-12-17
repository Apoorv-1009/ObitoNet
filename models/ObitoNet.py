import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from utils.logging import *
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
import random
import numpy as np
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

from transformers import AutoImageProcessor, ViTModel, ViTConfig


torch.autograd.set_detect_anomaly(True)

## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # This is the learnable weight matrix W
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) # This is the learnable weight matrix W'
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: torch.Tensor: (B, N, C)
        Returns:
            torch.Tensor: (B, N, C)

        B = Batch size
        N = Number of tokens
        C = Embedding dimension

        Note:
            Self Attention mechanism for the Transformer model
        """

        # Get the shape of input tensor
        B, N, C = x.shape

        # Learnable Linear transformation for Q, K, V, x: (B, N, C) -> (B, N, 3C)
        qkv = self.qkv(x)

        # Reshape to (3, B, num_heads, N, C // num_heads)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # Split into Queries (q), Keys (k), and Values (v)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # Attention matrix computation
        attn = (q @ k.transpose(-2, -1)) * self.scale # QK^T / sqrt(d_k)
        attn = attn.softmax(dim=-1) # Softmax(QK^T / sqrt(d_k))
        attn = self.attn_drop(attn) # Dropout

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # Softmax(QK^T / sqrt(d_k)) * V
        x = self.proj(x) # Learnable transformation
        x = self.proj_drop(x) # Dropout

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        """
        Args:
            x: torch.Tensor: (B, N, C)
        Returns:
            torch.Tensor: (B, N, C)

        B = Batch size
        N = Number of tokens
        C = Embedding dimension
        """
        # NOTE: drop path can be removed to help overfitting to the data
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

class MAEEncoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, tokens, pos):
        # transformer
        tokens = self.blocks(tokens, pos)
        tokens = self.norm(tokens)
        return tokens

class Embedder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class PCTokenizer (nn.Module):
    def __init__(self, config, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.mask_type = config.transformer_config.mask_type
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Embedder(encoder_channel = self.encoder_dims)
        self.trans_dim = config.transformer_config.trans_dim
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G
    
    def forward(self, xyz, noaug = False):
        neighborhood, center = self.group_divider(xyz)
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        tokens = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        return tokens, pos, bool_masked_pos, center, neighborhood
        
    def group_divider(self, xyz):
            '''
                input: B N 3
                ---------------------------
                output: B G M 3
                center : B G 3
            '''
            batch_size, num_points, _ = xyz.shape
            # fps the centers out
            center = self.fps(xyz, self.num_group) # B G 3
            # knn to get the neighborhood
            _, idx = self.knn(xyz, center) # B G M
            assert idx.size(1) == self.num_group
            assert idx.size(2) == self.group_size
            idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
            idx = idx + idx_base
            idx = idx.view(-1)
            neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
            neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
            # normalize
            neighborhood = neighborhood - center.unsqueeze(2)
            return neighborhood, center
    
    def fps(self, data, number):
        '''
            data B N 3
            number int
        '''
        # print(number)
        # print("yoyoyo",data.scalar_type())
        fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
        fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return fps_data

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Defined as Identity as a placeholder for the head, can be changed to Learnable Transformation
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_token_num):
        """
        Args:
            x: torch.Tensor: (B, N, C)
        
        Returns:
            x: torch.Tensor: (B, N, C)

        Note:
            Encoder only Transformer model, which has 'depth' number of Encoder blocks
        """

        # Runs 'depth' times
        for i, block in enumerate(self.blocks):
            # print("i: ", i)
            # 
            x = block(x)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the first N tokens
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim=384, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.W_x = nn.Linear(dim, dim * 3, bias=qkv_bias) # This is the learnable weight matrix W_x
        self.W_y = nn.Linear(dim, dim * 3, bias=qkv_bias) # This is the learnable weight matrix W_y

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) # This is the learnable weight matrix W'
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor: (B, N, C)
            y: torch.Tensor: (B, M, C)

        Returns:
            torch.Tensor: (B, N, C)

        B = Batch size
        N = Number of tokens
        M = Number of tokens
        C = Embedding dimension

        Note:
            Cross Attention between two sets of tokens x and y
            Query Tensors taken from x
            Key and Value Tensors taken from y
        """

        # Get the shape of input tensor 
        B, N, C = x.shape 
        _, M, _ = y.shape 

        # Learnable Linear transformation for K, V, x: (B, N, C) -> (B, N, 3C)
        x_qkv = self.W_x(x)
        y_qkv = self.W_y(y)

        # Reshape to (3, B, num_heads, N, C // num_heads)
        x_qkv = x_qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        y_qkv = y_qkv.reshape(B, M, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # Split into Queries (q), Keys (k), and Values (v)
        # q, k_,v_ = x_qkv[0], x_qkv[1], x_qkv[2]
        # q_, k, v = y_qkv[0], y_qkv[1], y_qkv[2]

        q, k_,v_ = torch.unbind(x_qkv, dim=0)
        q_, k, v = torch.unbind(y_qkv, dim=0)

        # Attention matrix computation
        attn = (q @ k.transpose(-2, -1)) * self.scale # QK^T / sqrt(d_k)
        attn = attn.softmax(dim=-1) # Softmax(QK^T / sqrt(d_k))
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C) # Softmax(QK^T / sqrt(d_k)) * V
        x_attn = self.proj(x_attn) # Learnable transformation

        return x_attn

class ObitoNetCA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.trans_dim = config.transformer_config.trans_dim

        # Cross Attention
        self.cross_attn = CrossAttention(dim=self.trans_dim)

        #MAE Decoder
        self.decoder_depth = config.transformer_config.decoder_depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = MAEDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        #Reconstruction head
        self.group_size = config.group_size
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 45*self.group_size, 1)
        )
    
    def load_model_from_ckpt(self, ckpt_path):
        """
        Loads pretrained weights into the Point_MAE model for continued training.

        Args:
            ckpt_path (str): Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file is not found.
        """
        if ckpt_path is not None:
            print_log(f"Loading checkpoint from {ckpt_path}...", logger="Point_MAE")
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")  # Load checkpoint
                state_dict = ckpt['model'] if 'model' in ckpt else ckpt

                # Load state_dict into the model
                incompatible = self.load_state_dict(state_dict, strict=False)

                # Log missing and unexpected keys
                if incompatible.missing_keys:
                    print_log(f"Missing keys: {get_missing_parameters_message(incompatible.missing_keys)}", logger="Point_MAE")
                if incompatible.unexpected_keys:
                    print_log(f"Unexpected keys: {get_unexpected_parameters_message(incompatible.unexpected_keys)}", logger="Point_MAE")

                print_log(f"Checkpoint successfully loaded from {ckpt_path}", logger="Point_MAE")
            except FileNotFoundError:
                print_log(f"Checkpoint file not found: {ckpt_path}", logger="Point_MAE")
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        else:
            print_log("No checkpoint path provided. Training from scratch.", logger="Point_MAE")
            self.apply(self._init_weights)  # Initialize weights if no checkpoint is provided

    def forward(self, pc_tokens, N, img_tokens, **kwargs):
        # Apply Cross Attention to pointcloud and image tokens
        tokens = self.cross_attn(pc_tokens, img_tokens)

        # Pass all embeddings through Masked Decoder
        tokens_recreated = self.MAE_decoder(tokens, N)

        # Pass through reconstruction head
        B, M, C = tokens_recreated.shape
        pts_reconstructed = self.increase_dim(tokens_recreated.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
        return pts_reconstructed, B, M 

class ObitoNetPC(nn.Module):
    def __init__(self, config):
        super().__init__()
        #Point Cloud Tokenizer
        self.group_size = config.group_size
        self.num_groups = config.num_group
        self.masked_pc_tokenizer = PCTokenizer(config, self.num_groups, self.group_size)

        #Point Cloud Masked AutoEncoder
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MAEEncoder(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
    
    def load_model_from_ckpt(self, ckpt_path):
        """
        Loads pretrained weights into the Point_MAE model for continued training.

        Args:
            ckpt_path (str): Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file is not found.
        """
        if ckpt_path is not None:
            print_log(f"Loading checkpoint from {ckpt_path}...", logger="Point_MAE")
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")  # Load checkpoint
                state_dict = ckpt['model'] if 'model' in ckpt else ckpt

                # Load state_dict into the model
                incompatible = self.load_state_dict(state_dict, strict=False)

                # Log missing and unexpected keys
                if incompatible.missing_keys:
                    print_log(f"Missing keys: {get_missing_parameters_message(incompatible.missing_keys)}", logger="Point_MAE")
                if incompatible.unexpected_keys:
                    print_log(f"Unexpected keys: {get_unexpected_parameters_message(incompatible.unexpected_keys)}", logger="Point_MAE")

                print_log(f"Checkpoint successfully loaded from {ckpt_path}", logger="Point_MAE")
            except FileNotFoundError:
                print_log(f"Checkpoint file not found: {ckpt_path}", logger="Point_MAE")
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        else:
            print_log("No checkpoint path provided. Training from scratch.", logger="Point_MAE")
            self.apply(self._init_weights)  # Initialize weights if no checkpoint is provided

    def forward(self, pts, **kwargs):
        # Extract tokens, and their knn position
        # feature_emb_vis: Unmasked descriptor of the point cloud from PointNet++
        # pos: xyz position of the knn cluster (learned)
        # bool_masked_pos: Masked index position to be used before Encoding
        # center: xyz position of the center of the knn cluster
        # neighborhood: xyz position of all knn clusters
        feature_emb_vis, feature_pos, bool_masked_pos, center, neighborhood = self.masked_pc_tokenizer(pts)
        
        # Pass visible tokens through the Masked Encoder 
        feature_emb_vis = self.MAE_encoder(feature_emb_vis, feature_pos)
        feature_emb_vis, mask = feature_emb_vis, bool_masked_pos
        
        ##### Black Box #####
        B,_,C = feature_emb_vis.shape # B VIS C

        # Pass masks through linear layer
        pos_emb_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emb_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _,N,_ = pos_emb_mask.shape
        tokens_masked = self.mask_token.expand(B, N, -1)
        #####################
        
        # Concatenate the embeddings
        feature_emb_all = torch.cat([feature_emb_vis, tokens_masked], dim=1)
        pos_all = torch.cat([pos_emb_vis, pos_emb_mask], dim=1)

        # Create tokens: add feature embeddings and positional encoding
        tokens = feature_emb_all + pos_all

        # Print shape of feature_emb_all and pos_all
        # print("feature_emb_all.shape: ", feature_emb_all.shape) # [1, 32, 384]
        # print("pos_all.shape: ", pos_all.shape) # [1, 32, 384]

        # Return tokens, number of tokens
        return tokens, N, mask, center, neighborhood

class ObitoNetImg(nn.Module):
    """
    Image Encoder for ObitoNet
    Extracts image tokens from images
    """
    def __init__(self, config):
        super().__init__()

        self.patch_size = int(((config.transformer_config.image_dim)**2 / config.num_group)**0.5)
        print("Patch Size: ", self.patch_size)
        # Load ViT model configuration
        self.configuration = ViTConfig(hidden_size=config.transformer_config.trans_dim,
                                       patch_size=self.patch_size,
                                       )

        # Load the image_processor
        self.image_processor = AutoImageProcessor.from_pretrained(config.transformer_config.vit_model_name,
                                                                  use_fast=True
                                                                  )

        # Load the ViT model
        self.vit_model = ViTModel(config=self.configuration)


    def load_model_from_ckpt(self, ckpt_path):
        """
        Loads pretrained weights into the Point_MAE model for continued training.

        Args:
            ckpt_path (str): Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file is not found.
        """
        if ckpt_path is not None:
            print_log(f"Loading checkpoint from {ckpt_path}...", logger="Point_MAE")
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")  # Load checkpoint
                state_dict = ckpt['model'] if 'model' in ckpt else ckpt

                # Load state_dict into the model
                incompatible = self.load_state_dict(state_dict, strict=False)

                # Log missing and unexpected keys
                if incompatible.missing_keys:
                    print_log(f"Missing keys: {get_missing_parameters_message(incompatible.missing_keys)}", logger="Point_MAE")
                if incompatible.unexpected_keys:
                    print_log(f"Unexpected keys: {get_unexpected_parameters_message(incompatible.unexpected_keys)}", logger="Point_MAE")

                print_log(f"Checkpoint successfully loaded from {ckpt_path}", logger="Point_MAE")
            except FileNotFoundError:
                print_log(f"Checkpoint file not found: {ckpt_path}", logger="Point_MAE")
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        else:
            print_log("No checkpoint path provided. Training from scratch.", logger="Point_MAE")
            self.apply(self._init_weights)  # Initialize weights if no checkpoint is provided
    
    def forward(self, img):
        
        # Preprocess the image
        img_processed = self.image_processor(img, return_tensors="pt")

        # Pass through the ViT model
        output = self.vit_model(**img_processed)

        # Extract image_tokens without CLS token
        image_tokens = output.last_hidden_state[:, 1:, :]

        return image_tokens
      
class ObitoNet(nn.Module):
    def __init__(self, config, ObitoNetPC, ObitoNetImg, ObitoNetCA):
        super().__init__()

        # Point Cloud Encoder
        self.obitonet_pc = ObitoNetPC

        # Image Encoder
        self.obitonet_img = ObitoNetImg

        # Cross Attention Decoder
        self.obitonet_ca = ObitoNetCA

        #Training Loss
        self.trans_dim = config.transformer_config.trans_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

        self.num_groups = config.num_group        

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def load_model_from_ckpt(self, ckpt_path):
        """
        Loads pretrained weights into the Point_MAE model for continued training.

        Args:
            ckpt_path (str): Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file is not found.
        """
        if ckpt_path is not None:
            print_log(f"Loading checkpoint from {ckpt_path}...", logger="Point_MAE")
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")  # Load checkpoint
                state_dict = ckpt['model'] if 'model' in ckpt else ckpt

                # Load state_dict into the model
                incompatible = self.load_state_dict(state_dict, strict=False)

                # Log missing and unexpected keys
                if incompatible.missing_keys:
                    print_log(f"Missing keys: {get_missing_parameters_message(incompatible.missing_keys)}", logger="Point_MAE")
                if incompatible.unexpected_keys:
                    print_log(f"Unexpected keys: {get_unexpected_parameters_message(incompatible.unexpected_keys)}", logger="Point_MAE")

                print_log(f"Checkpoint successfully loaded from {ckpt_path}", logger="Point_MAE")
            except FileNotFoundError:
                print_log(f"Checkpoint file not found: {ckpt_path}", logger="Point_MAE")
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        else:
            print_log("No checkpoint path provided. Training from scratch.", logger="Point_MAE")
            self.apply(self._init_weights)  # Initialize weights if no checkpoint is provided

    def forward(self, pts, img, vis = False, **kwargs):
        """
        Args:
            pts: torch.Tensor: (B, N, 3)
            img: torch.Tensor: (B, C, H, W)
            vis: bool: Visualization flag

        Returns:
            torch.Tensor: (B, N, 3)

        B = Batch size
        N = Number of points
        C = Number of channels
        H = Height
        W = Width
        """
        # Extract encoded PC tokens from the ObitoNet Point Cloud Arm
        pc_tokens, N, mask, center, neighborhood  = self.obitonet_pc(pts)

        # Ensure img is in the correct shape (B, C, H, W)
        if img.shape[-1] == 3:  # If last dimension is the channel
            img = img.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)

        # Extract encoded img tokens from the ObitoNet Image Arm
        img_tokens = self.obitonet_img(img)

        pts_reconstructed, B, M = self.obitonet_ca(pc_tokens, N, img_tokens)

        # Extract ground truth points
        gt_points = neighborhood[mask].reshape(B*M,-1,3)
        loss1 = self.loss_func(pts_reconstructed, gt_points)

        if vis: #visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_groups - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = pts_reconstructed + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1
        