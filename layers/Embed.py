import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class ClusterTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ClusterTokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class ClusterEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(ClusterEmbedding, self).__init__()

        self.value_embedding = ClusterTokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        return self.dropout(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DisDataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DisDataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


# 块特征提取模块
class BlockFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入块，形状为 (num_blocks, block_len)
        Returns:
            features (torch.Tensor): 特征矩阵，形状为 (num_blocks, 2)
        """
        mu = x.mean(dim=-1)  # 均值
        sigma = x.std(dim=-1)  # 标准差
        return torch.stack([mu, sigma], dim=-1)


# 动态重要性评分模块
class ImportanceScorer(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, 8),  # 降维层
            nn.ReLU(),
            nn.Linear(8, 1)  # 重要性分数输出
        )

    def forward(self, features):
        scores = self.scorer(features)  # (num_blocks, 1)
        return torch.sigmoid(scores)  # 归一化到[0,1]

class PatchEmbeddingR(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout, reverse, n_clusters=5):
        super(PatchEmbeddingR, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.reverse = reverse
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # 使用神经网络学习位置嵌入
        self.position_embedding = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 混合策略参数
        self.high_importance_ratio = 0.5  # 初始高重要性块翻转比例
        self.low_importance_ratio = 0.5  # 初始低重要性块翻转比例

        # 特征提取与聚类翻转
        self.feature_extractor = BlockFeatureExtractor()
        self.cluster_flip = ClusterFlipModule(n_clusters=n_clusters)

    def compute_flip_indices_novel(self, x):
        """
        使用更高级的方法生成反转索引，基于特征的聚类或注意力分布。
        """
        # 计算每个补丁的特征均值
        feature_mean = x.mean(dim=-1)  # (batch_size * n_vars, num_patches)

        # 对均值进行聚类（例如使用 K-Means 聚类）
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(feature_mean.cpu().numpy())

        # 选择属于特定聚类（例如，高均值聚类）的索引
        high_cluster = clusters == clusters.max()
        all_indices = torch.tensor(high_cluster.nonzero()[0], device=x.device)

        # 限制反转的序列数为总体的十分之一
        num_to_flip = max(1, len(all_indices) // 10)  # 至少保留一个索引
        flip_indices = all_indices[:num_to_flip]

        return flip_indices

    def compute_flip_indices_mixed(self, x, epoch=1, max_epochs=1):
        """
        使用混合策略生成翻转索引
        Args:
            x (torch.Tensor): 输入patch，形状为 (num_patches, patch_len)
            epoch (int): 当前训练epoch
            max_epochs (int): 最大训练epoch数
        Returns:
            flip_indices (torch.Tensor): 需要翻转的patch索引
        """
        num_patches = x.shape[0]
        importance_scores = self.importance_scorer(x)  # (num_patches,)

        # 动态调整翻转比例
        if epoch is not None and max_epochs is not None:
            progress = epoch / max_epochs
            self.high_importance_ratio = 0.5 * (1 - progress)  # 逐渐减少高重要性块翻转比例
            self.low_importance_ratio = 0.5 * progress  # 逐渐增加低重要性块翻转比例

        # 选择高重要性块
        num_high = max(1, int(self.high_importance_ratio * num_patches))
        _, high_indices = torch.topk(importance_scores, num_high)  # 选择重要性最高的块

        # 选择低重要性块
        num_low = max(1, int(self.low_importance_ratio * num_patches))
        _, low_indices = torch.topk(-importance_scores, num_low)  # 选择重要性最低的块

        # 合并索引
        flip_indices = torch.cat([high_indices, low_indices])
        return flip_indices

    def forward(self, x, reverse, epoch=1, max_epochs=1):
        # do patching
        n_vars = x.shape[1]

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        if reverse:
            # # 选择指定位置翻转
            # flip_indexs = random.sample(range(0, x.shape[0] // 2), x.shape[0] // 6)
            # for idx in flip_indexs:
            #     x[idx] = torch.flip(x[idx], [1])
            # 使用基于聚类方法生成翻转索引
            # flip_indices = self.compute_flip_indices_novel(x)
            # flip_indices = self.compute_flip_indices_mixed(x, epoch, max_epochs)

            x_feature = self.feature_extractor(x)
            x = self.cluster_flip(x_feature, x, epoch=epoch, max_epochs=max_epochs)
            # for idx in flip_indices:
            #     x[idx] = torch.flip(x[idx], [1])  # 对时间维度进行翻转

        # Input encoding
        # x = self.value_embedding(x) + self.position_embedding(x)
        x = self.value_embedding(x)
        position_emb = self.position_embedding(x)
        x = x + position_emb

        return self.dropout(x), n_vars

# 聚类与翻转模块
class ClusterFlipModule(nn.Module):
    def __init__(self, n_clusters=3):
        super().__init__()
        self.n_clusters = n_clusters
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, 2))  # 可学习的聚类中心
        self.importance_scorer = ImportanceScorer()
        # 混合策略参数
        self.high_importance_ratio = 0.5  # 初始高重要性块翻转比例
        self.low_importance_ratio = 0.5  # 初始低重要性块翻转比例

    def forward(self, features, blocks, epoch, max_epochs):
        """
        Args:
            features (torch.Tensor): 特征矩阵 (num_blocks, 2)
            blocks (torch.Tensor): 原始块数据 (num_blocks, block_len)
        Returns:
            flipped_blocks (torch.Tensor): 翻转后的块
        """

        num_patches = blocks.shape[0]
        # importance_scores = self.importance_scorer(blocks)  # (num_patches,)

        # 动态聚类
        distances = torch.cdist(features, self.cluster_centers)  # (num_blocks, n_clusters)
        labels = torch.argmin(distances, dim=-1)  # (num_blocks,)

        # 簇内翻转策略
        flipped_blocks = blocks.clone()
        for cluster_id in range(self.n_clusters):
            mask = (labels == cluster_id)
            cluster_blocks = blocks[mask]

            if len(cluster_blocks) > 0:
                # 计算簇内重要性分数
                cluster_features = features[mask]
                importance = self.importance_scorer(cluster_features)

                # 动态调整翻转比例
                if epoch is not None and max_epochs is not None:
                    progress = epoch / max_epochs
                    self.high_importance_ratio = 0.5 * (1 - progress)  # 逐渐减少高重要性块翻转比例
                    self.low_importance_ratio = 0.5 * progress  # 逐渐增加低重要性块翻转比例

                # 选择高重要性块
                num_high = max(1, int(self.high_importance_ratio * num_patches))
                _, high_indices = torch.topk(importance.squeeze(), num_high)  # 选择重要性最高的块

                # 选择低重要性块
                num_low = max(1, int(self.low_importance_ratio * num_patches))
                _, low_indices = torch.topk(-importance.squeeze(), num_low)  # 选择重要性最低的块

                # 合并索引
                flip_indices = torch.cat([high_indices, low_indices])
                for idx in flip_indices:
                    flipped_blocks[mask][idx] = torch.flip(flipped_blocks[mask][idx], [-1])

        return flipped_blocks

        #         # 选择重要性最低的25%进行翻转
        #         flip_num = max(1, int(0.25 * len(cluster_blocks)))
        #         _, least_important = torch.topk(-importance.squeeze(), flip_num)
        #
        #         for idx in least_important:
        #             flipped_blocks[mask][idx] = torch.flip(flipped_blocks[mask][idx], [-1])
        #
        # return flipped_blocks

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

# class PatchIEmbedding(nn.Module):
#     def __init__(self, d_model, patch_len, stride, dropout):
#         super(PatchEmbedding, self).__init__()
#         # Patching
#         self.patch_len = patch_len
#         self.stride = stride
#         self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
#
#         # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
#         self.value_embedding = TokenEmbedding(patch_len, d_model)
#
#         # Positional embedding
#         self.position_embedding = PositionalEmbedding(d_model)
#
#         # Residual dropout
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         # do patching
#         n_vars = x.shape[1]
#
#         x = self.padding_patch_layer(x)
#         x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
#         x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
#
#         # Input encoding
#         x = self.value_embedding(x) + self.position_embedding(x)
#         return self.dropout(x), n_vars


class PatchIEmbedding(nn.Module):
    """子序列分割与嵌入"""

    def __init__(self, d_model, patch_len, stride, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """输入: [batch, num_vars, seq_len]"""
        batch, num_vars, seq_len = x.shape

        # 分割子序列
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2)  # [batch, vars, patch_len, num_patches]

        # 嵌入投影
        patches = self.projection(patches.reshape(-1, self.patch_len))  # [batch*vars*num_patches, d_model]
        patches = patches.reshape(batch, num_vars, num_patches, -1)

        return self.dropout(patches).reshape(batch * num_vars, num_patches, -1), num_vars

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)