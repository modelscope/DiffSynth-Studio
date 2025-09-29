# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleAdapter(nn.Module):
    """
    將 Δ 向量（由 S-P 的影像特徵差）映射成：
      (A) 每層的 logits 偏置向量 b_q, b_k（用於 rank-1 外積加到 attn logits）
      (B) 每層的 KV style tokens（可選：更強，但稍進階）
    以 zero-init 結尾層確保初次不改變模型行為。
    """
    def __init__(self, delta_dim: int, d_model: int, n_layers: int, n_heads: int,
                 kv_tokens: int = 0):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.kv_tokens = kv_tokens  # 若 >0 則啟用 KV tokens 注入

        hidden = max(256, delta_dim // 2)

        # 產生每層、每頭用的 b_q, b_k（簡化為每層每頭一條向量，再在時間上 broadcast）
        self.to_bq = nn.Sequential(
            nn.Linear(delta_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, n_layers * n_heads)
        )
        self.to_bk = nn.Sequential(
            nn.Linear(delta_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, n_layers * n_heads)
        )
        # zero-init 讓初期不影響模型
        nn.init.zeros_(self.to_bq[-1].weight); nn.init.zeros_(self.to_bq[-1].bias)
        nn.init.zeros_(self.to_bk[-1].weight); nn.init.zeros_(self.to_bk[-1].bias)

        # 可選：KV style tokens（每層 M 個，維度 d_model）
        if kv_tokens > 0:
            self.to_kv_tokens = nn.Sequential(
                nn.Linear(delta_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, n_layers * kv_tokens * d_model)
            )
            nn.init.zeros_(self.to_kv_tokens[-1].weight)
            nn.init.zeros_(self.to_kv_tokens[-1].bias)

    def forward(self, delta: torch.Tensor):
        """
        delta: [B, delta_dim]（通常 B=1）
        回傳：
          bq, bk: list of length n_layers；每個 tensor 形狀 [B, n_heads]
          kv_tokens: list of length n_layers；每個 tensor [B, kv_tokens, d_model] 或 None
        """
        B, D = delta.shape
        bq = self.to_bq(delta).view(B, self.n_layers, self.n_heads)
        bk = self.to_bk(delta).view(B, self.n_layers, self.n_heads)

        kv_tokens = None
        if self.kv_tokens > 0:
            kv = self.to_kv_tokens(delta).view(
                B, self.n_layers, self.kv_tokens, self.d_model
            )
            kv_tokens = [kv[:, i] for i in range(self.n_layers)]
        bq_list = [bq[:, i] for i in range(self.n_layers)]
        bk_list = [bk[:, i] for i in range(self.n_layers)]
        return bq_list, bk_list, kv_tokens
