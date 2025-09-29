# -*- coding: utf-8 -*-
import types
import torch
import torch.nn.functional as F

class LogitsBiasHook:
    """
    在 cross-attn 的 forward 內，對 attn logits 加上 rank-1 偏置：
      logits += lambda_style * (bq.unsqueeze(-1) @ bk.unsqueeze(-2))  (per-head)
    其中 bq, bk 由 StyleAdapter 產生，會在時間/長度維度上 broadcast。
    """
    def __init__(self, bq_list, bk_list, lambda_style: float = 0.5):
        self.bq_list = bq_list
        self.bk_list = bk_list
        self.lambda_style = lambda_style
        self.layer_idx = 0  # 掃描時用，按發現順序對應到第幾層

    def wrap_attn(self, attn_module):
        if hasattr(attn_module, "_wrapped_by_logits_bias"):
            return  # 避免重複包

        old_forward = attn_module.forward

        def new_forward(module, query, key, value, attn_mask=None, **kwargs):
            out = old_forward(query, key, value, attn_mask=attn_mask, **kwargs)
            # 嘗試取得 attn_weights（不同實作可能返回 (out, weights) 或只返回 out）
            if isinstance(out, tuple) and len(out) == 2:
                attn_out, attn_weights = out
            else:
                # 如果拿不到權重，就不處理（相容性保護）
                return out

            # 取對應層的 bq, bk  -> 形狀 [B, n_heads]
            B = attn_weights.shape[0]
            H = attn_weights.shape[1]
            Lq = attn_weights.shape[2]
            Lk = attn_weights.shape[3]
            if self.layer_idx >= len(self.bq_list):
                return out
            bq = self.bq_list[self.layer_idx]  # [B, H]
            bk = self.bk_list[self.layer_idx]  # [B, H]
            if bq.shape[1] != H or bk.shape[1] != H:
                # head 數不匹配就跳過
                return out

            # rank-1 外積 -> [B, H, Lq, Lk]（對時間/空間 broadcast）
            bias = (bq.unsqueeze(-1) * bk.unsqueeze(-1))  # [B, H, 1]
            bias = bias.unsqueeze(-1)  # [B, H, 1, 1]
            bias = bias.expand(B, H, Lq, Lk)  # broadcast

            attn_weights = attn_weights + self.lambda_style * bias
            self.layer_idx += 1
            return attn_out, attn_weights

        attn_module.forward = types.MethodType(new_forward, attn_module)
        attn_module._wrapped_by_logits_bias = True


class KVTokensHook:
    """
    在 cross-attn 的 forward 內，把 style tokens 拼到 K/V 後面。
    kv_tokens_per_layer: list of tensors [B, M, d_model]
    注意：這依賴 attn 模組內部用線性投影到多頭，若 attn.forward 接口只吃 (q,k,v)，
    我們只能在外層包一層前置投影；此處做最小假設：可以在 forward 入口直接 concat K/V。
    """
    def __init__(self, kv_tokens_per_layer, lambda_tokens: float = 1.0):
        self.kv_tokens_per_layer = kv_tokens_per_layer
        self.lambda_tokens = lambda_tokens
        self.layer_idx = 0

    def wrap_attn(self, attn_module):
        if self.kv_tokens_per_layer is None:
            return
        if hasattr(attn_module, "_wrapped_by_kv_tokens"):
            return

        old_forward = attn_module.forward

        def new_forward(module, query, key, value, *args, **kwargs):
            # 如果該層沒有 style tokens，直接走原函式
            if self.layer_idx >= len(self.kv_tokens_per_layer) or \
               self.kv_tokens_per_layer[self.layer_idx] is None:
                return old_forward(query, key, value, *args, **kwargs)

            kv_tok = self.kv_tokens_per_layer[self.layer_idx]  # [B_tok, M, d_model]
            self.layer_idx += 1

            # 對齊 batch、dtype、device
            Bq = query.shape[0]
            Bt = kv_tok.shape[0]
            if Bt == 1 and Bq > 1:
                kv_tok = kv_tok.expand(Bq, -1, -1)  # 複用同一組 tokens
            elif Bt != Bq:
                # 保底：裁切或重複到 Bq（通常不會走到）
                kv_tok = kv_tok[:1].expand(Bq, -1, -1)

            kv_tok = kv_tok.to(dtype=key.dtype, device=key.device)

            # 在序列維度拼接（[B, L, C]）
            key2   = torch.cat([key,   kv_tok], dim=1)
            value2 = torch.cat([value, self.lambda_tokens * kv_tok], dim=1)

            # ★ 重要：用位置參數呼叫，別傳 attn_mask / 其他 kwargs ★
            return old_forward(query, key2, value2)

        attn_module.forward = types.MethodType(new_forward, attn_module)
        attn_module._wrapped_by_kv_tokens = True
