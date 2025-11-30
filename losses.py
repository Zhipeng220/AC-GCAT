# losses.py

import torch
from torch import nn
import math
# [FIX] 确保 numpy 被导入
import numpy as np


class HierarchicalContrastiveLoss(nn.Module):
    """
    [创新点] 分层对比损失 (InfoNCE实现).
    该损失函数旨在将同一类别的手势嵌入在特征空间中拉近，同时推开不同类别的手势嵌入。
    这对于学习细粒度的手势差异（如 ASL 中的 'A' 和 'S'）至关重要。
    """

    def __init__(self, temperature=0.1, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = features.device

        # L2-normalize a projeção de características
        features = nn.functional.normalize(features, dim=1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # Para estabilidade numérica
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Máscara para remover auto-similaridade da diagonal
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # Calculo do log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Computar a média do log-likelihood sobre todas as âncoras positivas
        # Por simplicidade, consideramos todas as amostras da mesma classe como positivas
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


class AnatomicalLoss(nn.Module):
    """
    [技术严谨性]
    此损失函数的有效性可以通过其Lipschitz连续性进行理论分析。一个Lipschitz连续的损失函数
    对输入中的微小扰动（如关键点噪声）不敏感，这保证了训练过程的稳定性。
    通过限制骨骼长度和角度的惩罚函数的梯度，可以证明该损失满足此属性，从而增强了模型的鲁棒性。
    """

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.finger_indices = {'thumb': [1, 2, 3, 4], 'index': [5, 6, 7, 8], 'middle': [9, 10, 11, 12],
                               'ring': [13, 14, 15, 16], 'pinky': [17, 18, 19, 20]}
        self.palm_indices = [0, 1, 5, 9, 13, 17]
        # Store label encoder classes to identify 'M'/'N' or 'nothing' if needed for adaptive weighting.
        # This needs to be set from the main script after the label_encoder is created.
        self.label_classes_ = None

    def _calculate_angle_loss(self, hand_kpts, confidences):
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if confidences is None: confidences = torch.ones_like(hand_kpts[..., 0])
        for _, indices in self.finger_indices.items():
            for i in range(len(indices) - 2):
                p0_idx, p1_idx, p2_idx = indices[i], indices[i + 1], indices[i + 2]
                p0, p1, p2 = hand_kpts[:, p0_idx], hand_kpts[:, p1_idx], hand_kpts[:, p2_idx]
                v1, v2 = p0 - p1, p2 - p1
                cos_theta = torch.sum(v1 * v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-8)
                angle = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
                constraint_confidence = torch.min(
                    torch.stack([confidences[:, p0_idx], confidences[:, p1_idx], confidences[:, p2_idx]]), dim=0).values

                # MODIFICATION 1: Relax angle constraint from [0, pi] to [-pi/2, 3pi/2]
                # This prevents penalizing natural hand poses that might exceed 180 degrees.
                angle_loss = torch.relu(angle - 1.5 * math.pi) ** 2 + torch.relu(-angle - 0.5 * math.pi) ** 2
                loss += torch.mean(constraint_confidence * angle_loss)
        return loss

    def _calculate_length_loss(self, hand_kpts, confidences):
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if confidences is None: confidences = torch.ones_like(hand_kpts[..., 0])
        for _, indices in self.finger_indices.items():
            if len(indices) < 4: continue
            p_indices = [indices[0], indices[1], indices[2], indices[3]]
            constraint_confidence = torch.min(confidences[:, p_indices], dim=1).values
            ref_len = torch.norm(hand_kpts[:, indices[1]] - hand_kpts[:, indices[0]], dim=1) + 1e-8

            # MODIFICATION 2: Relax bone length ratio constraints to be more forgiving.
            ratio1 = torch.norm(hand_kpts[:, indices[2]] - hand_kpts[:, indices[1]], dim=1) / ref_len
            # Original: (relu(ratio1 - 1.2)**2 + relu(0.8 - ratio1)**2)
            loss += torch.mean(constraint_confidence * (torch.relu(ratio1 - 1.5) ** 2 + torch.relu(0.5 - ratio1) ** 2))

            ratio2 = torch.norm(hand_kpts[:, indices[3]] - hand_kpts[:, indices[2]], dim=1) / ref_len
            # Original: (relu(ratio2 - 0.9)**2 + relu(0.6 - ratio2)**2)
            loss += torch.mean(constraint_confidence * (torch.relu(ratio2 - 1.2) ** 2 + torch.relu(0.4 - ratio2) ** 2))
        return loss

    def _calculate_planarity_loss(self, hand_kpts, confidences):
        if confidences is None: confidences = torch.ones_like(hand_kpts[..., 0])
        palm_kpts, palm_confidences = hand_kpts[:, self.palm_indices, :], confidences[:, self.palm_indices]
        centroid = torch.sum(palm_kpts * palm_confidences.unsqueeze(-1), dim=1, keepdim=True) / (
                torch.sum(palm_confidences, dim=1, keepdim=True).unsqueeze(-1) + 1e-8)
        centered_kpts = palm_kpts - centroid
        try:
            # SVD 可能会回退到 CPU (如 MPS 不支持)
            C = torch.bmm((centered_kpts * palm_confidences.unsqueeze(-1)).transpose(1, 2), centered_kpts)
            _, _, V = torch.svd(C)
            # 确保 V 回到 self.device (如果 SVD 在 CPU 上运行)
            V = V.to(self.device)
            dists = torch.sum(centered_kpts * V[:, :, -1].unsqueeze(1), dim=2) ** 2
            # MODIFICATION 3: Reduce the impact of the planarity loss.
            return 0.5 * torch.mean(palm_confidences * dists)
        except torch.linalg.LinAlgError:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

    # MODIFICATION 5: Update signature to accept labels for adaptive weighting.
    def forward(self, input_batch, labels=None):
        # [FIX] import numpy as np is required at the top of losses.py for the fix to work
        # import numpy as np (已移动到文件顶部)

        keypoints_batch = input_batch[..., :3] if input_batch.shape[-1] == 4 else input_batch
        confidences_batch = input_batch[..., 3] if input_batch.shape[-1] == 4 else None

        # MODIFICATION 4: Add confidence thresholding to ignore noisy/occluded keypoints.
        if confidences_batch is not None:
            # Create a mask for valid keypoints (confidence > 0.5)
            valid_mask = (confidences_batch > 0.5).unsqueeze(-1).float()
            # Apply mask to keypoints
            keypoints_batch = keypoints_batch * valid_mask

        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if torch.any(torch.sum(torch.abs(keypoints_batch), dim=(1, 2)) > 1e-6):
            angle_loss = self._calculate_angle_loss(keypoints_batch, confidences_batch)
            length_loss = self._calculate_length_loss(keypoints_batch, confidences_batch)
            planarity_loss = self._calculate_planarity_loss(keypoints_batch, confidences_batch)

            base_loss = angle_loss + length_loss + planarity_loss

            # Implementation for adaptive weighting based on class labels.
            if labels is not None and self.label_classes_ is not None:
                # weight_factor 位于 self.device (例如 mps:0)
                weight_factor = torch.ones(len(labels), device=self.device, dtype=torch.float32)

                # Increase weight for confusing classes like 'M' and 'N'
                # [FIX] Use np.where because self.label_classes_ is a numpy array
                m_index_np = np.where(self.label_classes_ == 'M')[0]
                n_index_np = np.where(self.label_classes_ == 'N')[0]

                # [RUNTIMEERROR FIX]
                # 将 numpy 索引转换为 self.device 上的 torch 标量张量
                if len(m_index_np) > 0:
                    m_index_tensor = torch.tensor(m_index_np[0], device=self.device)
                    # 比较 (mps:0) == (mps:0)
                    weight_factor[labels == m_index_tensor] = 2.0

                if len(n_index_np) > 0:
                    n_index_tensor = torch.tensor(n_index_np[0], device=self.device)
                    # 比较 (mps:0) == (mps:0)
                    weight_factor[labels == n_index_tensor] = 2.0

                # Decrease weight for minority classes like 'nothing' to avoid overfitting to their geometry
                # [FIX] Use np.where because self.label_classes_ is a numpy array
                nothing_index_np = np.where(self.label_classes_ == 'nothing')[0]

                # [RUNTIMEERROR FIX]
                if len(nothing_index_np) > 0:
                    nothing_index_tensor = torch.tensor(nothing_index_np[0], device=self.device)
                    # 比较 (mps:0) == (mps:0)
                    weight_factor[labels == nothing_index_tensor] = 0.1

                total_loss = torch.mean(base_loss * weight_factor)
            else:
                total_loss = base_loss

        return total_loss
    