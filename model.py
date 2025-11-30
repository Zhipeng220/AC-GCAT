# model.py
import torch
from torch import nn
import torchvision.models as models  # Keep import just in case


# --- [ _FFN, FourierPositionalEncoding, MLPBaseline, AdvancedAttentionBlock, AdvancedAttentionModel, GCNLayer - UNCHANGED ] ---
class _FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout)
        )

    def forward(self, x): return self.main(x)


class FourierPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_freqs=10, alpha=10000.0):
        super().__init__()
        self.embed_dim, self.num_freqs, self.alpha = embed_dim, num_freqs, alpha
        self.freq_bands = nn.Parameter(alpha ** (torch.linspace(0.0, 1.0, num_freqs) ** 2), requires_grad=False)
        self.projection = nn.Linear(3 * 2 * num_freqs, embed_dim)  # Only uses 3D coords

    def forward(self, coords):  # Expects coords with shape (B, N, 3)
        B, N, _ = coords.shape
        mean = torch.mean(coords, dim=1, keepdim=True)
        std = torch.std(coords, dim=1, keepdim=True) + 1e-7
        norm_coords = (coords - mean) / std
        scaled_coords = norm_coords.unsqueeze(-1) * self.freq_bands
        fourier_features = torch.cat([torch.sin(scaled_coords), torch.cos(scaled_coords)], dim=-1)
        return self.projection(fourier_features.view(B, N, -1))


class MLPBaseline(nn.Module):
    def __init__(self, input_size, num_classes):  # input_size = 21 * input_dim (e.g., 21*11)
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.5)
        )
        self.fc = nn.Linear(256, num_classes)

    # --- [MODIFIED] 更新签名以匹配 train.py 的调用 ---
    def forward(self, x_keypoints, x_image_features):  # 忽略 x_image_features
        x = x_keypoints # 仅使用关键点
        features = self.main(x.view(x.size(0), -1))
        return self.fc(features), features
    # --- [MODIFIED END] ---


class AdvancedAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = _FFN(embed_dim, embed_dim * 4, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class AdvancedAttentionModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=4, num_heads=8, embed_dim=256, dropout=0.2):
        super().__init__()
        self.coord_embed = nn.Linear(input_dim, embed_dim)  # input_dim is engineered (e.g., 11)
        self.pos_encoder = FourierPositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([AdvancedAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim * 21, num_classes)  # Flattened features

    # --- [MODIFIED] 更新签名以匹配 train.py 的调用 ---
    def forward(self, x_keypoints, x_image_features):  # 忽略 x_image_features
        x = x_keypoints # 仅使用关键点
        # Only use first 3 dims for pos encoding
        features = self.dropout(self.coord_embed(x) + self.pos_encoder(x[..., :3]))
        for block in self.blocks: features = block(features)
        embeddings = self.final_norm(features).view(features.size(0), -1)  # Flatten
        logits = self.fc(embeddings)
        return logits, embeddings
    # --- [MODIFIED END] ---


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj):
        aggregated_features = torch.einsum('bnn,bnj->bnj', adj, x)
        return self.linear(aggregated_features)


# --- [ NEW: ImageOnlyBaseline for Ablation Study ] ---
class ImageOnlyBaseline(nn.Module):
    def __init__(self, image_feature_dim, num_classes, dropout=0.5):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(image_feature_dim, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    # 匹配 Hybrid 模型的输入签名
    def forward(self, x_keypoints, x_image_features):
        logits = self.main(x_image_features)
        return logits, x_image_features  # 返回 (logits, embeddings)


# --- [ NEW: End ] ---


# --- [ Modify HybridAttentionModel ] ---
class HybridAttentionModel(nn.Module):
    # --- [ MODIFIED: 添加 use_cross_attn 和 use_gcn_branch 参数 ] ---
    def __init__(self, input_dim, num_classes, num_layers=4, num_heads=8, embed_dim=256, dropout=0.2,
                 use_gated_fusion=True, use_dynamic_graph=True,
                 image_feature_dim=2048,
                 use_cross_attn=True, use_gcn_branch=True): # <--- 这就是您需要的那一行
        # --- [ MODIFIED: End ] ---
        super().__init__()
        self.embed_dim = embed_dim
        self.image_feature_dim = image_feature_dim

        # --- [ NEW: 存储消融标志 ] ---
        self.use_gated_fusion = use_gated_fusion
        self.use_dynamic_graph = use_dynamic_graph
        self.use_cross_attn = use_cross_attn
        self.use_gcn_branch = use_gcn_branch
        # --- [ NEW: End ] ---

        # --- [ Keypoint branch - Self-Attention ] ---
        self.coord_embed = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = FourierPositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_blocks = nn.ModuleList(
            [AdvancedAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.attention_norm = nn.LayerNorm(embed_dim)

        # --- [ Keypoint branch - GCN (Conditional) ] ---
        if self.use_gcn_branch:
            self.gcn_blocks = nn.ModuleList()
            gcn_dims = [embed_dim] + [embed_dim] * 2
            for i in range(len(gcn_dims) - 1):
                self.gcn_blocks.append(GCNLayer(gcn_dims[i], gcn_dims[i + 1]))
                self.gcn_blocks.append(nn.ReLU())
                self.gcn_blocks.append(nn.LayerNorm(gcn_dims[i + 1]))
            self.register_buffer('adj_static', self.create_adjacency_matrix())
            if self.use_dynamic_graph:
                self.graph_q_proj = nn.Linear(embed_dim, embed_dim // 4)
                self.graph_k_proj = nn.Linear(embed_dim, embed_dim // 4)
                self.graph_activation = nn.Tanh()

        # --- [ Fusion Layers (Conditional) ] ---
        if self.use_cross_attn:
            self.cross_attn_norm_kpt = nn.LayerNorm(embed_dim)
            self.img_k_proj = nn.Linear(self.image_feature_dim, embed_dim)
            self.img_v_proj = nn.Linear(self.image_feature_dim, embed_dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.cross_attn_dropout = nn.Dropout(dropout)

        # --- [ Final FC (Conditional Input Size) ] ---
        keypoint_fusion_dim = embed_dim * 21  # (Att_flat)

        if self.use_gated_fusion:
            # Gating layer input dim depends on whether GCN is used
            gating_input_dim = keypoint_fusion_dim * 2 if self.use_gcn_branch else keypoint_fusion_dim
            self.gating_layer = nn.Sequential(
                nn.Linear(gating_input_dim, gating_input_dim // 4), nn.ReLU(),
                nn.Linear(gating_input_dim // 4, keypoint_fusion_dim), nn.Sigmoid()  # Output dim is Att_flat
            )
            final_fc_input_dim = keypoint_fusion_dim  # Gated output
        else:
            # No gating, input dim depends on whether GCN is used
            final_fc_input_dim = keypoint_fusion_dim * 2 if self.use_gcn_branch else keypoint_fusion_dim

        # If NOT using cross-attn, we must add image features (Late Fusion)
        if not self.use_cross_attn:
            final_fc_input_dim += self.image_feature_dim

        self.fc = nn.Linear(final_fc_input_dim, num_classes)

    def create_adjacency_matrix(self):
        # ... (UNCHANGED)
        adj = torch.zeros(21, 21)
        connections = [[0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
                       [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8],
                       [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16],
                       [17, 18], [18, 19], [19, 20]]
        for i, j in connections: adj[i, j] = adj[j, i] = 1
        adj = adj + torch.eye(21)
        d_inv_sqrt = torch.pow(adj.sum(1), -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        return torch.mm(torch.diag(d_inv_sqrt), adj) @ torch.diag(d_inv_sqrt)

    def forward(self, x_keypoints, x_image_features):
        B = x_keypoints.size(0)

        # --- Keypoint processing ---
        coord_embedding = self.coord_embed(x_keypoints)
        pos_encoding = self.pos_encoder(x_keypoints[..., :3])
        embedded = self.dropout(coord_embedding + pos_encoding)

        # 1. Keypoint Self-Attention
        attn_features = embedded
        for block in self.attention_blocks: attn_features = block(attn_features)

        # --- [ MODIFIED: Conditional GCN Branch ] ---
        if self.use_gcn_branch:
            gcn_features = embedded
            if self.use_dynamic_graph:
                q = self.graph_q_proj(embedded);
                k = self.graph_k_proj(embedded)
                adj_dynamic = self.graph_activation(torch.einsum('bif,bjf->bij', q, k))
                fused_adj = self.adj_static.unsqueeze(0) + adj_dynamic
            else:
                fused_adj = self.adj_static.unsqueeze(0).expand(B, -1, -1)
            for module in self.gcn_blocks:
                if isinstance(module, GCNLayer):
                    gcn_features = module(gcn_features, fused_adj)
                elif isinstance(module, nn.LayerNorm):
                    gcn_features = module(gcn_features)
                else:
                    gcn_features = module(gcn_features)
            gcn_flat = gcn_features.view(B, -1)
        else:
            # If no GCN, create a zero tensor to keep fusion logic consistent
            gcn_flat = torch.zeros(B, self.embed_dim * 21, device=x_keypoints.device)
        # --- [ MODIFIED: End ] ---

        # --- [ MODIFIED: Conditional Cross-Attention Fusion ] ---
        if self.use_cross_attn:
            q_kpt = self.cross_attn_norm_kpt(attn_features)
            k_img = self.img_k_proj(x_image_features).unsqueeze(1)
            v_img = self.img_v_proj(x_image_features).unsqueeze(1)
            fused_attn_features, _ = self.cross_attn(q_kpt, k_img, v_img)
            fused_attn_features = attn_features + self.cross_attn_dropout(fused_attn_features)
            attn_flat = self.attention_norm(fused_attn_features).view(B, -1)
        else:
            # Use original self-attention features
            attn_flat = self.attention_norm(attn_features).view(B, -1)
        # --- [ MODIFIED: End ] ---

        # --- [ MODIFIED: Conditional Gated Fusion ] ---
        gate = None
        if self.use_gated_fusion:
            # Gate input depends on GCN branch
            gate_input = torch.cat([attn_flat, gcn_flat], dim=1) if self.use_gcn_branch else attn_flat
            gate = self.gating_layer(gate_input)

            # Gating logic - if no GCN, gate only combines Attn with (1-gate)*0
            gcn_term = (1 - gate) * gcn_flat if self.use_gcn_branch else 0
            fused_keypoints = gate * attn_flat + gcn_term
        else:
            # No gating, just concatenate
            fused_keypoints = torch.cat([attn_flat, gcn_flat], dim=1) if self.use_gcn_branch else attn_flat
        # --- [ MODIFIED: End ] ---

        # --- [ MODIFIED: Conditional Classification ] ---
        if self.use_cross_attn:
            logits = self.fc(fused_keypoints)
            final_embeddings = fused_keypoints
        else:
            # Late Fusion: concatenate image features *after* fusion
            combined_features = torch.cat([fused_keypoints, x_image_features], dim=1)
            logits = self.fc(combined_features)
            final_embeddings = combined_features

        return logits, final_embeddings, gate
    # --- [ Forward method END ] ---
# --- [ Modify HybridAttentionModel END ] ---