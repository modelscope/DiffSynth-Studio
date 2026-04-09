# WanVideoPipeline Technical Analysis

這份文件是對 `DiffSynth-Studio/diffsynth/pipelines/wan_video.py` 的詳細代碼解析。該文件定義了 `WanVideoPipeline`，這是 DiffSynth-Studio 中用於處理 Wan2.1 視頻生成模型的核心管道。

## 1. 核心類別設計

### 1.1 `WanVideoPipeline` (第 31 行)
這是主要的控制類別，繼承自 `BasePipeline`。
*   **初始化 (`__init__`)**:
    *   負責初始化各種子模型：`text_encoder`, `image_encoder`, `dit` (主要生成模型), `vae`, `motion_controller`, `vace` (上下文嵌入), `vap`, `animate_adapter` 等。
    *   定義了 `units` 列表：這是此 Pipeline 的核心特色，它將複雜的生成過程拆解為一系列獨立的 "Unit" (單元)，按順序執行。
*   **載入模型 (`from_pretrained`)**:
    *   支援從 ModelConfig 自動下載並載入模型。
    *   **包含重要的重定向邏輯 (Redirection)**：自動將舊的 `.pth` 檔名映射到新的 `.safetensors` 路徑 (第 110-124 行)。
*   **執行推理 (`__call__`)**:
    *   定義了所有可能的輸入參數（Prompt, Image, Video, ControlNet, VACE, Animate 等）。
    *   **Unit Runner 機制**：透過 `self.unit_runner` 依次執行 `self.units` 中的每個單元，逐步構建 `inputs_shared` (共享數據), `inputs_posi` (正向條件), `inputs_nega` (負向條件)。
    *   **Denoising Loop (去噪循環)**：
        *   加載模型到 GPU (`load_models_to_device`).
        *   迭代 Timesteps。
        *   支援動態切換模型 (`switch_DiT_boundary`)：可以在去噪過程中切換主模型 (例如從 Base 模型切換到 Refiner)。
        *   呼叫 `self.model_fn` 進行單步推理。
        *   使用 Scheduler 更新 Latents。
    *   **解碼 (Decode)**：最後使用 VAE 將 Latents 解碼回視頻像素。

## 2. Pipeline Units (模組化處理單元)
代碼採用了模組化設計，每個 Unit 負責處理特定類型的輸入或功能。

### 2.1 基礎處理單元
*   `WanVideoUnit_ShapeChecker`: 確保輸入的長寬高符合模型要求 (16x16 對齊)。
*   `WanVideoUnit_NoiseInitializer`: 初始化隨機噪聲。
*   `WanVideoUnit_PromptEmbedder`: 處理文本提示詞，使用 Text Encoder 生成 Context。

### 2.2 圖像/視頻嵌入單元
*   `WanVideoUnit_InputVideoEmbedder`:
    *   處理輸入視頻 (Video-to-Video)。
    *   使用 VAE Encode 輸入視頻並加入噪聲 (加噪過程)。
    *   處理 `vace_reference_image` 的編碼。
*   `WanVideoUnit_ImageEmbedderCLIP`: 使用 CLIP Image Encoder 處理輸入圖像特徵。
*   `WanVideoUnit_ImageEmbedderVAE`: 使用 VAE 處理圖像條件 (用於 Image-to-Video)。

### 2.3 控制與功能單元
*   `WanVideoUnit_VACE` (第 617 行): **Context Embedder 核心邏輯**
    *   處理 `vace_video` (影片條件), `vace_video_mask`, `depth_video` (深度圖)。
    *   **Masking 處理**: 將輸入分為 `inactive` (背景/不變部分) 和 `reactive` (前景/變化部分)，分別編碼後拼接。
    *   **拼接邏輯**: 將 Reference Image Latents, Video Latents, Mask Latents, Depth Latents 全部 concat 起來成為 `vace_context`。
    *   **112->96 Channel 相容**: 如果模型是舊版 112 ch 但現在只要 96 ch，會做特殊的 padding 處理 (第 687 行)。
*   `WanVideoUnit_FunControl`: 處理 ControlNet 視頻輸入。
*   `WanVideoUnit_FunCameraControl`: 處理相機運鏡控制 (Camera Pose)。
*   `WanVideoUnit_SpeedControl`: 處理速度控制 (Motion Bucket ID)。
*   `WanVideoUnit_TeaCache`: 實現 TeaCache 加速算法，跳過部分 Transformer 步驟以加速推理。

### 2.4 動畫與高級功能單元
*   `WanVideoUnit_AnimatePoseLatents` / `WanVideoUnit_AnimateFacePixelValues`: 處理人體姿態和臉部特徵，用於 Character Animation。
*   `WanVideoUnit_VAP` (Video-As-Prompt): 處理視頻提示詞功能。
*   `WanVideoUnit_LongCatVideo`: 處理長視頻生成的特殊邏輯。

## 3. 推理函數 (`model_fn_wan_video`)
這是實際執行 DiT (Diffusion Transformer) 前向傳播的函數 (第 1143 行)。

*   **Temporal Tiling**: 支援 `TemporalTiler_BCTHW`，允許處理超長視頻，透過滑動窗口 (Sliding Window) 分段推理並混合結果。
*   **LongCat & S2V 分支**: 針對 LongCat 模型和 Speech-to-Video 模型有專門的處理分支。
*   **Embedding 準備**:
    *   計算 Timestep Embedding (`t_mod`)。
    *   處理 Motion Controller 的調節。
    *   合併 Text Context 和 CLIP Feature。
    *   處理 VAE Image Embedding (`y`)。
*   **Patchify**: 將 Latents 切分成 Patch 序列。
*   **VACE 注入**:
    *   如果存在 `vace_context`，會通過 `vace` 模型 (Context Embedder) 計算特徵 (`vace_hints`)。
    *   **關鍵**: 在 Transformer Block 循環中，將 VACE hints 加到主特徵 `x` 上 (`x = x + current_vace_hint * vace_scale`) (第 1391 行)。
*   **Transformer Block Loop**:
    *   迭代執行 DiT Blocks。
    *   支援 Gradient Checkpointing (節省顯存)。
    *   支援 Sequence Parallel (多卡並行)。
    *   整合了 VAP, VACE, Animate Adapter 等額外模組的注入邏輯。
*   **Head & Unpatchify**: 最後通過 Output Head 並還原成圖像/視頻格式。

## 4. 數據流總結
1.  **使用者輸入** -> `__call__`
2.  **預處理** -> `units` (Resize, VAE Encode, Text Encode)
3.  **條件準備** -> `inputs_shared` (Latents, Embeddings, Conditions)
4.  **去噪循環** -> 
    *   `model_fn` -> 
    *   `DiT Forward` (結合 VACE/ControlNet/Adapter) -> 
    *   `Scheduler Step`
5.  **解碼** -> `VAE Decode` -> **輸出視頻**
