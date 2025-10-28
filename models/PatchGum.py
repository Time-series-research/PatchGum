import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchIEmbedding

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8, topk_ratio=0.1):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.topk_ratio = topk_ratio
        self.patch_len = patch_len
        self.stride = stride

        # 子序列分割与嵌入
        self.patch_embedding = PatchIEmbedding(
            configs.d_model, patch_len, stride, configs.dropout)

        # 动态评分模块
        self.scoring_network = nn.Sequential(
            nn.Linear(configs.d_model, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )

        # 完整编码器
        self.full_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # 轻量特征提取器
        self.lite_extractor = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.GELU(),
            nn.Linear(configs.d_model // 2, configs.d_model),
            nn.Dropout(configs.dropout)
        )

        # 重建模块
        self.rebuild_layer = nn.Linear(configs.d_model, patch_len)

        # 预测头
        self._init_task_specific_head(configs)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        输入:
        x_enc: [batch_size, seq_len, num_vars]
        输出: 根据任务类型的预测结果
        """
        # 子序列嵌入 [batch*num_vars, num_patches, d_model]
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        batch_size = x_enc.shape[0]
        x_enc = x_enc.permute(0, 2, 1)  # [batch, num_vars, seq_len]
        patches, n_vars = self.patch_embedding(x_enc)
        num_patches = patches.size(1)

        # 动态重要性评分 [batch*num_vars, num_patches]
        scores = self.scoring_network(patches).squeeze(-1)

        # 可微分Top-k选择
        k = max(1, int(num_patches * self.topk_ratio))
        topk_indices = self._gumbel_topk(scores, k, training=self.training)

        # 随机Top-k
        # probs = torch.ones(scores.shape)  # 均匀分布
        # topk_indices = torch.multinomial(probs, k, replacement=False)

        # 创建处理掩码
        full_mask = self._create_sparse_mask(topk_indices, (batch_size * n_vars, num_patches))
        lite_mask = ~full_mask

        # 分组处理
        full_feat = self._process_group(patches, full_mask, self.full_encoder)
        lite_feat = self._process_group(patches, lite_mask, self.lite_extractor)

        # 特征合并
        combined = torch.zeros_like(patches)
        combined[full_mask] = full_feat
        combined[lite_mask] = lite_feat

        # 重建序列
        combined = self.rebuild_layer(combined)  # [batch*num_vars, num_patches, patch_len]
        output = self._reconstruct_sequence(combined, batch_size, n_vars)
        # output = self._reconstruct_sequence_random(combined, batch_size, n_vars)

        dec_out = output.permute(0, 2, 1)
        dec_out = self._handle_task(dec_out, x_enc)

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # 任务特定处理
        return dec_out
        # return dec_out.permute(0, 2, 1)

    def _process_group(self, patches, mask, processor):
        """处理指定组的子序列"""
        flat_patches = patches.flatten(0, 1)  # [total_patches, d_model]
        selected = flat_patches[mask.flatten()]

        # 处理并解包返回值
        processed_output = processor(selected.unsqueeze(0))

        # 处理多返回值情况
        if isinstance(processed_output, tuple):
            processed = processed_output[0]  # 取主输出
        else:
            processed = processed_output

        return processed.squeeze(0)

    def _reconstruct_sequence(self, patches, batch_size, n_vars):
        """将子序列重建为完整序列"""
        patches = patches.reshape(batch_size, n_vars, -1, self.patch_len)
        patches = patches.permute(0, 1, 3, 2)  # [batch, vars, patch_len, num_patches]
        seq_len = (patches.shape[-1] - 1) * self.stride + self.patch_len
        output = torch.zeros((batch_size, n_vars, seq_len), device=patches.device)

        # 重叠相加重建
        for i in range(patches.shape[-1]):
            start = i * self.stride
            end = start + self.patch_len
            output[..., start:end] += patches[..., i]
        return output

    def _reconstruct_sequence_random(self, patches, batch_size, n_vars):
        """
        将 patch 表示重建为完整序列，支持任意预测长度 pred_len
        假设 patch 是等距滑窗切分的，使用重叠相加方式重建
        """
        # 重新变形: [B * n_vars, num_patches, patch_len] -> [B, n_vars, patch_len, num_patches]
        patches = patches.reshape(batch_size, n_vars, -1, self.patch_len).permute(0, 1, 3, 2)
        num_patches = patches.shape[-1]

        # 输出序列长度由预测长度 pred_len 决定
        seq_len = self.pred_len
        output = torch.zeros((batch_size, n_vars, seq_len), device=patches.device)
        overlap_counter = torch.zeros((batch_size, n_vars, seq_len), device=patches.device)

        for i in range(num_patches):
            start = i * self.stride
            end = start + self.patch_len
            if start >= seq_len:
                break
            if end > seq_len:
                end = seq_len
                patch = patches[..., :end - start, i]  # 截断 patch 超出部分
            else:
                patch = patches[..., :, i]
            output[..., start:end] += patch
            overlap_counter[..., start:end] += 1

        # 避免除以0，平均重叠区域
        output = output / torch.clamp(overlap_counter, min=1.0)
        return output

    def _gumbel_topk(self, scores, k, training=True, tau=0.1):
        """可微分Top-k选择"""
        if training:
            gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10))
            scores = scores + gumbel
        return torch.topk(scores, k, dim=1).indices

    def _create_sparse_mask(self, indices, shape):
        """创建稀疏掩码矩阵"""
        mask = torch.zeros(shape, dtype=torch.bool, device=indices.device)
        batch_indices = torch.arange(shape[0], device=indices.device)[:, None]
        mask[batch_indices, indices] = True
        return mask

    def _init_task_specific_head(self, configs):
        """初始化任务特定头部"""
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.head = nn.Linear(configs.enc_in * configs.seq_len, configs.pred_len)
        elif self.task_name == 'classification':
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(configs.enc_in * configs.seq_len, 256),
                nn.ReLU(),
                nn.Linear(256, configs.num_class)
            )

    def _handle_task(self, output, x_enc):
        """任务特定处理"""
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return output[:, -self.pred_len:, :]
        elif self.task_name == 'classification':
            return self.head(output.flatten(1))
        # elif self.task_name == 'imputation':
        #     return output * mask + x_enc * (~mask)
        return output
