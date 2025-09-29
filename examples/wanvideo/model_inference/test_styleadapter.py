# -*- coding: utf-8 -*-
import os, torch, cv2, numpy as np
from PIL import Image

from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

# >>> NEW: 引入 open_clip / adapter / hooks
import open_clip
from torchvision import transforms
from style_adapter import StyleAdapter
from attn_hooks import LogitsBiasHook, KVTokensHook

import inspect

# =========================
# 你的基本設定（略）...
# =========================

# >>> NEW: 簡單的 CLIP 特徵抽取（ViT-L/14@336）
def build_clip(device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.to(device)
    model.eval()
    return model, preprocess

@torch.inference_mode()
def img_feat(clip_model, preprocess, img: Image.Image, device="cuda"):
    x = preprocess(img).unsqueeze(0).to(device)
    feat = clip_model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat  # [1, D]

# >>> NEW: 掃描模型，抓 cross-attn 模組（名稱匹配法，Wan通常會有包含'attn'字樣的模組）
# def iter_cross_attn_modules(dit):
#     # 優先逐 block 掃描，命中 self_attn / cross_attn 名稱
#     for bi, block in enumerate(getattr(dit, "blocks", [])):
#         for name, m in block.named_modules():
#             lname = name.lower()
#             if ("attn" in lname or "attention" in lname) and hasattr(m, "forward"):
#                 yield f"blocks[{bi}].{name}", m
#             print("[hook]", name)

# def register_style_hooks(pipeline, bq_list, bk_list, kv_tokens_per_layer=None,
#                          lambda_style=0.5, lambda_tokens=1.0, max_layers=18):
#     root = pipeline.dit
#     lb = LogitsBiasHook(bq_list[:max_layers], bk_list[:max_layers], lambda_style)
#     kv = KVTokensHook(kv_tokens_per_layer[:max_layers] if kv_tokens_per_layer else None, lambda_tokens)

#     n = 0
#     for name, m in iter_cross_attn_modules(root):
#         if n >= max_layers: break
#         lb.wrap_attn(m)
#         if kv_tokens_per_layer is not None:
#             kv.wrap_attn(m)
#         n += 1
#     print(f"[StyleHooks] Attached to {n} attention modules under dit.blocks (max_layers={max_layers}).")

def get_inner_qkv_attentions(dit):
    """
    直接取每個 block 的 self_attn.attn / cross_attn.attn 內層注意力模組。
    不做簽名檢查，確保一定命中。
    """
    modules = []
    blocks = getattr(dit, "blocks", [])
    for bi, block in enumerate(blocks):
        # self_attn.attn
        if hasattr(block, "self_attn") and hasattr(block.self_attn, "attn"):
            modules.append((f"blocks[{bi}].self_attn.attn", block.self_attn.attn))
        # cross_attn.attn
        if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "attn"):
            modules.append((f"blocks[{bi}].cross_attn.attn", block.cross_attn.attn))
    return modules

def register_style_hooks(pipeline, bq_list, bk_list, kv_tokens_per_layer=None,
                         lambda_style=0.0, lambda_tokens=1.0, max_layers=18):
    root = getattr(pipeline, "dit", None)
    assert root is not None, "WanVideoPipeline 沒有 dit。"

    # 先用 KV tokens（穩定、與返回權重無關）；logits-bias 先關掉
    kv = KVTokensHook(kv_tokens_per_layer[:max_layers] if kv_tokens_per_layer else None,
                      lambda_tokens)

    targets = get_inner_qkv_attentions(root)[:max_layers]
    n = 0
    for name, m in targets:
        kv.wrap_attn(m)
        print("[hook-qkv]", name)
        n += 1
    print(f"[StyleHooks] Attached to {n} qkv-attention modules (max_layers={max_layers}).")


# >>> 你的 main 入口
if __name__ == "__main__":
    device = "cuda"

    # 1) 建立 pipeline（略）
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B",
                        origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()

    # 2) 載入你的 P 與 S，算 Δ
    PLATE_IMG = "/eva_data0/lynn/VideoGAI/plate_0.png"
    STYLIZED_PLATE_IMG = "/eva_data0/lynn/VideoGAI/P018_VPWIP_029_0150_Styleframe001_TargetStyle_Frame.1009.png"
    FIRST_IMG = "/eva_data0/lynn/VideoGAI/P018_VPWIP_029_0150_Styleframe001_TargetStyle_Frame.1009.png"

    clip_model, preprocess = build_clip(device)
    imgP = Image.open(PLATE_IMG).convert("RGB")
    imgS = Image.open(STYLIZED_PLATE_IMG).convert("RGB")
    imgF = Image.open(FIRST_IMG).convert("RGB")
    fP = img_feat(clip_model, preprocess, imgP, device)  # [1, D]
    fS = img_feat(clip_model, preprocess, imgS, device)  # [1, D]
    delta = (fS - fP)  # [1, D]
    delta = delta / (delta.norm(dim=-1, keepdim=True) + 1e-6)

    # 3) 構建 StyleAdapter，產生每層的風格信號
    #    d_model / n_layers / n_heads 可用保守值；若不確定，n_heads=8~16 OK，會自動 broadcast
    D_DELTA = delta.shape[-1]
    style_adapter = StyleAdapter(
        delta_dim=D_DELTA,
        d_model=1536,   # 對齊 dim
        n_layers=18,    # 只打前 18 層
        n_heads=12,     # 對齊 head 數
        kv_tokens=4     # 先用 4 個 style tokens；VRAM 夠可拉到 8
    ).to(device)

    bq_list, bk_list, kv_tokens_per_layer = style_adapter(delta)

    register_style_hooks(
        pipe,
        bq_list=bq_list, bk_list=bk_list,
        kv_tokens_per_layer=kv_tokens_per_layer,  # 這版我們主用 KV tokens
        lambda_style=0.0,       # 先不開 logits-bias（很多 attn 不回傳權重）
        lambda_tokens=1,
        max_layers=18
    )

    # 5) 準備控制（例如全域正規化的 depth clip）與 style 參考（仍可給 S）
    TARGET_H, TARGET_W = 480, 832
    FPS = 24
    DEPTH_NORM_MP4 = "/eva_data0/lynn/VideoGAI/DiffSynth-Studio/depth_0030.mp4"
    control_video = VideoData(DEPTH_NORM_MP4, height=TARGET_H, width=TARGET_W)
    ref_img = imgF.resize((TARGET_W, TARGET_H))  # 仍可作為 vace_reference_image 的補強

    PROMPT = "保持输入视频中的原始场景布局和运动；将所提供的参考图像的艺术风格在整个片段中一致地应用；保留人脸特征和细节"
    NEG_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    # 6) 生成單一短 clip（無自回歸）
    video = pipe(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        vace_video=control_video,
        vace_reference_image=ref_img,   # 與 logits-bias 並用：S 作 style anchor + Δ 作細緻拉力
        seed=1,
        tiled=True,
        height=TARGET_H,
        width=TARGET_W,
    )
    from diffsynth import save_video
    save_video(video, "stylized_clip_with_delta_v4_0030.mp4", fps=FPS, quality=5)
    print("[Done] stylized_clip_with_delta_v3_0030.mp4")
