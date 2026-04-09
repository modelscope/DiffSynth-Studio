# VACE Depth Condition Implementation Analysis

這份文件詳細解析了為了在 VACE (Context Embedder) 中加入 **Depth Video** 作為條件輸入所做的修改，以及訓練時的權重更新策略。

## 1. 架構修改權總覽 (Architecture Changes)

原有的 VACE 輸入主要包含「原影片 latent」和「Mask latent」。為了加入深度資訊，我們在輸入層級進行了擴充：

*   **原始輸入**: `Video Latents (16ch)` + `Mask Latents (16ch)` = **32 channels** (邏輯上)
    *   *註：實際代碼中 VAE Encode 出來通常是 16ch，mask 經過處理也是對應維度。具體維度在下文代碼解析中詳述。*
*   **新輸入**: `Video Latents` + `Mask Latents` + `Depth Video Latents`
*   **影響範圍**: 
    *   Pipeline 預處理邏輯 (`WanVideoUnit_VACE`)
    *   VACE 模型的輸入投影層 (`vace_patch_embedding`)

## 2. 代碼修改解析 (`diffsynth/pipelines/wan_video.py`)

核心修改位於 `WanVideoUnit_VACE` 類別的 `process` 方法中。

### 2.1 接收 Depth Video 輸入
在 `__call__` 和 `WanVideoUnit_VACE` 中新增了 `depth_video` 參數。

```python
# diffsynth/pipelines/wan_video.py

class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            # 新增 "depth_video" 到 input_params
            input_params=(..., "depth_video", ...), 
            ...
        )
```

### 2.2 Depth 預處理與 VAE Encoding
如同處理 RGB 影片一樣，深度影片也需要經過 VAE 壓縮到 Latent Space。

```python
if depth_video is not None:
    # 1. 預處理: 確保格式正確 (B, C, T, H, W)
    depth_video = pipe.preprocess_video(depth_video)
    
    # 2. VAE Encode: 將像素空間的 Depth Map 轉為 Latent 
    # VAE 輸出維度通常為 16 channels
    depth_video_latents = pipe.vae.encode(depth_video, ...).to(...)
```

### 2.3 特徵拼接 (Concatenation)
這是最關鍵的步驟。代碼將所有條件在 Channel 維度 (`dim=1`) 進行拼接。

```python
# 1. 基礎拼接: Video (Inactive + Reactive) + Mask
vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)

# 2. 加入 Depth
if depth_video is not None:
     vace_context = torch.concat((vace_context, depth_video_latents), dim=1)
```

**維度計算範例**:
*   `vace_video_latents`: 32 channels (包含 inactive 和 reactive 兩部分，各 16ch)
*   `vace_mask_latents`: 64 channels (經過特殊處理和 padding)
*   `depth_video_latents`: 16 channels
*   **總輸入維度**: 32 + 64 + 16 = **112 channels**

### 2.4 自動 Padding 機制 (兼容性處理)
代碼中包含一段重要的邏輯，用於處理模型輸入維度不匹配的情況。這解釋了為什麼有時候輸入是 96ch，有時候是 112ch。

```python
# 如果模型權重期待 112 channels (因為包含了 depth)，但目前的輸入只有 96 channels (沒給 depth video)
if pipe.vace.vace_patch_embedding.in_channels == 112 and vace_context.shape[1] == 96:
    # 自動補 16 個 channel 的零，讓模型能夠正常運作 (Zero Padding)
    zeros = torch.zeros((..., 16, ...))
    vace_context = torch.concat((vace_context, zeros), dim=1)
```
這意味著，訓練好支援 Depth 的模型 (112ch)，在推理時如果**不給 Depth**，系統會自動填零，仍然可以運作（但效果可能不如有給 depth 好）。

## 3. 訓練權重更新 (`examples/wanvideo/model_training/train.py`)

為了讓模型適應新的輸入條件（Depth），訓練時必須更新特定的權重。

### 3.1 必須更新的權重：Context Embedder
因為輸入 Channel 數量改變了（增加了 16ch），VACE 的第一層卷積層 (`vace_patch_embedding`) 的權重形狀與預訓練模型不同（或者需要學習如何處理這新增的 16ch 資訊）。

在 `train.py` 中有如下邏輯：

```python
if hasattr(self.pipe, "vace") and self.pipe.vace is not None:
    # 強制解凍 VACE 的 Patch Embedding 層 -> 進行 Full Finetune
    self.pipe.vace.vace_patch_embedding.requires_grad_(True)
    
    # 保持 Bias 固定 (通常是為了穩定性，可選)
    if self.pipe.vace.vace_patch_embedding.bias is not None:
        self.pipe.vace.vace_patch_embedding.bias.requires_grad_(False)
```

### 3.2 建議的訓練策略 (Training Strategy)

1.  **Context Embedder (`vace_patch_embedding`)**:
    *   **狀態**: **Unfrozen (Full Finetune)**
    *   **原因**: 輸入維度物理改變，且必須從頭學習如何將 Depth Latent 映射到 VACE 的隱藏層空間。

2.  **VACE Transformer Blocks (`vace_blocks`)**:
    *   **狀態**: **LoRA Finetune** (或 Frozen，視數據量而定)
    *   **原因**: 當 Embedder 處理好輸入後，Transformer Blocks 接收到的隱藏層維度 (`dim=1536`) 是不變的。使用 LoRA 讓模型學習如何利用這些新的深度語義資訊即可，不需要破壞原本強大的預訓練權重。

3.  **其他模型 (DiT, VAE, T5)**:
    *   **狀態**: **Frozen**
    *   **原因**: 這些是生成的主幹，通常不需要為了適應新的 Condition 而從頭訓練。

## 4. 總結

要在 VACE 中加入 Depth Video，所做的修改如下：

1.  **Pipeline 層**: 修改 `WanVideoUnit_VACE`，在拼接 Latent 時將 Depth VAE Latent (16ch) 拼接到最後。
2.  **Model 層**: VACE 模型的輸入層 (`Conv3d`) 輸入通道數從 96 增加到 112。
3.  **Training 層**:
    *   **Full Finetune**: `vace.vace_patch_embedding` (適應新通道)。
    *   **LoRA Finetune**: `vace` 的其他部分 (學習深度與畫面的關聯)。
