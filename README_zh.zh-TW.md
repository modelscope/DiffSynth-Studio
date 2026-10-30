# DiffSynth-Studio

<a href="https://github.com/modelscope/DiffSynth-Studio"><img src=".github/workflows/logo.gif" title="Logo" style="max-width:100%;" width="55" /></a> <a href="https://trendshift.io/repositories/10946" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10946" alt="modelscope%2FDiffSynth-Studio | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a></p>

[![PyPI](https://img.shields.io/pypi/v/DiffSynth)](https://pypi.org/project/DiffSynth/)
[![license](https://img.shields.io/github/license/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/DiffSynth-Studio.svg)](https://GitHub.com/modelscope/DiffSynth-Studio/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/DiffSynth-Studio)](https://GitHub.com/modelscope/DiffSynth-Studio/commit/)
[![Discord](https://badgen.net//discord/members/Mm9suEeUDc)](https://discord.gg/Mm9suEeUDc)

[Switch to English](./README.md)

## 簡介

歡迎來到 Diffusion 模型的魔法世界！DiffSynth-Studio 是由[魔搭社群](https://www.modelscope.cn/)團隊開發和維護的開源 Diffusion 模型引擎。我們期望以框架建設孵化技術創新，凝聚開源社群的力量，探索生成式模型技術的豐富能力！

框架功能：

* [模型支援](#全部支援的模型)：框架集成了主流的開源 Diffusion 模型，涵蓋影像生成、影片生成、音訊生成，以及影像指標模型。
* [視訊記憶體管理](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Pipeline_Usage/VRAM_management.html)：在硬碟、記憶體、視訊記憶體之間動態排程模型引數，讓低視訊記憶體的消費級 GPU 也能執行大模型的推理。
* [引數量化](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Pipeline_Usage/Quantization.html)：將模型引數轉換為 NF4、INT8 等量化精度，大幅減少模型推理和 LoRA 模型訓練的視訊記憶體需求。
* [任意訓練](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Pipeline_Usage/Model_Training.html)：幾乎所有支援推理的模型都支援訓練，無論是基礎模型、LoRA，還是任何帶有額外輸入的 Adapter 模型。
* [拆分訓練](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Training/Split_Training.html)：使用計算圖推理引擎追蹤 Pipeline 中每個變數，將訓練過程拆分為兩階段，高效訓練。

參考資料：

* 開發者文件 (面向人類)：[中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/README.html)、[English version](https://diffsynth-studio-doc.readthedocs.io/en/latest/README.html)
* Agent Skills (面向 AI)：[DiffSynth-Studio Model Integration Skills](https://www.modelscope.cn/collections/DiffSynth-Studio/DiffSynth-Studio-Model-Integration-Skills)

檢視更多：

* [DiffSynth-ComfyUI](https://github.com/modelscope/DiffSynth-ComfyUI): 基於 DiffSynth-Studio 構建的 [ComfyUI](https://github.com/Comfy-Org/ComfyUI) 節點包，可在 ComfyUI 中構建 Workflow 執行模型推理。
* [DiffSynth-WebUI](https://github.com/modelscope/DiffSynth-WebUI): 基於 DiffSynth-Studio 構建的輕量化 LoRA 訓練工具，可在消費級 GPU 上訓練 LoRA 模型。
* [魔搭社群 AIGC 專區 (面向中國使用者)](https://modelscope.cn/aigc/home): 以 DiffSynth-Studio 為核心推理和訓練引擎的產品化功能，體驗開源模型生態的豐富潛力。
* [ModelScope Civision (for global users)](https://modelscope.ai/civision/home): Unlock the vast potential of the open-source model ecosystem through productized capabilities powered by DiffSynth-Studio.

## 更新歷史

> DiffSynth-Studio 經歷了大版本更新，部分舊功能已停止維護，如需使用舊版功能，請切換到大版本更新前的[最後一個歷史版本](https://github.com/modelscope/DiffSynth-Studio/tree/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3)。

> 目前本專案的開發人員有限，大部分工作由 [Artiprocher](https://github.com/Artiprocher) 和 [mi804](https://github.com/mi804) 負責，因此新功能的開發進展會比較緩慢，issue 的回覆和解決速度有限，我們對此感到非常抱歉，請各位開發者理解。

- **2026年10月21日** 我們已支援 Fibo 2.0，包含統一的文生圖與影像編輯推理、掩碼編輯、多參考影像編輯、文字渲染、低視訊記憶體推理，以及 [Fibo-2.0-dev](https://www.modelscope.cn/models/briaai/Fibo-2.0-dev) 在文生圖和影像編輯任務上的 LoRA 與全量訓練。詳情請參考[文件](/docs/zh/Model_Details/Fibo.md)和[示例程式碼](/examples/fibo/)。

- **2026年10月20日** 我們已支援 MOSS-Image，包含 [MOSS-Image-Base](https://modelscope.cn/models/openmoss/MOSS-Image-Base) 和 [MOSS-Image-SFT](https://modelscope.cn/models/openmoss/MOSS-Image-SFT) 的推理和訓練。另外，我們也提供了 [EntroPack 低位量化模型](https://www.modelscope.cn/models/DiffSynth-Studio/MOSS-Image-EntroPack)，最低支援 4G 視訊記憶體執行 3.5bit 的量化壓縮模型；以及通過 [OPD 蒸餾加速](https://diffsynth-studio-doc.readthedocs.io/en/latest/Training/Distillation.html)訓練的[加速模型](https://modelscope.cn/models/DiffSynth-Studio/MOSS-Image-SFT-TurboLoRA)，支援 8 步極速推理。

- **2026年10月16日** flux-3-image-klein 開源，歡迎加入影像生成模型家族！支援文生圖、多參考影像編輯、佈局控制、低視訊記憶體推理和 Base 訓練能力。詳情請參考[文件](/docs/zh/Model_Details/FLUX.3.md)和[示例程式碼](/examples/flux3/)。我們使用 [EntroPack](https://github.com/modelscope/entropack) 對模型進行了量化，提供了 3bit、4bit、6bit 幾個版本的預量化模型：[DiffSynth-Studio/flux-3-image-klein-entropack](https://www.modelscope.cn/models/DiffSynth-Studio/flux-3-image-klein-entropack)。

- **2026年10月12日** 我們已支援基於 OPD（On-Policy Distillation）的蒸餾加速訓練，現在能夠以極低的成本對模型進行蒸餾加速，詳見[蒸餾加速訓練](./docs/zh/Training/Distillation.md)。

- **2026年9月29日** 我們開源了 EntroPack——面向 PyTorch 的通用張量壓縮庫，支援逐位精確恢復的無失真壓縮和按目標位元速率控制體積的有失真壓縮，並提供 GPU 編解碼能力。它已作為量化後端接入 DiffSynth-Studio，可在推理與 LoRA 訓練中按任意目標位元速率壓縮模型權重。更多資訊請參考：

  * 專案主頁：https://github.com/modelscope/entropack
  * 技術報告：https://arxiv.org/abs/2609.34185
  * 預量化模型：https://www.modelscope.cn/collections/DiffSynth-Studio/EntroPack-Quantized-Models
  * 示例程式碼：[Z-Image-Turbo](/examples/z_image/model_inference_low_vram/Z-Image-Turbo-EntroPack.py)、[Qwen-Image-2.1](/examples/qwen_image_21/model_inference_low_vram/Qwen-Image-2.1-EntroPack.py)、[MiniMax-H3](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-EntroPack.py)、[LoRA 訓練示例](/examples/z_image/model_training/special/quant_training/Z-Image-Turbo-entropack_lossy_quant.sh)

- **2026年9月20日** Qwen-Image-2.1 開源，模型支援文生圖、影像編輯、帶透明通道的 RGBA 輸出，DiffSynth-Studio 已為其提供全面支援。詳情請參考[文件](/docs/zh/Model_Details/Qwen-Image-2.1.md)和[示例程式碼](/examples/qwen_image_21/)。

- **2026年9月14日** 我們訓練並開源了 DiffSynth-Music，基於 ACE-Step 的音樂可控生成模型，支援 Beats、Vocals、Accompany、Prosody、Reference 五種控制方式。更多資訊請參考：
    * 模型：https://modelscope.cn/models/DiffSynth-Studio/DiffSynth-Music
    * 技術報告：https://arxiv.org/abs/2609.12774
    * 程式碼：[./examples/diffsynth_music/](./examples/diffsynth_music/)

- **2026年9月11日** 我們已支援 YuE2，一個新的音樂生成模型。詳情請參考[文件](/docs/zh/Model_Details/YuE2.md)和[示例程式碼](/examples/yue2/)。

- **2026年9月10日** 我們接入了 [LTX-2.5](https://modelscope.cn/models/Lightricks/LTX-2.5)，這是 Lightricks 最新的音影片聯合生成模型。支援的功能包括自動時長預測的文生音影片、關鍵幀插值的圖生音影片、音訊驅動影片、音影片區域重生成、IC-LoRA 畫素空間上取樣、文生音訊、INT8 量化推理、低視訊記憶體推理以及模型訓練。詳情請參考[文件](/docs/zh/Model_Details/LTX-2.md)和[示例程式碼](/examples/ltx2/)。

- **2026年9月10日** 我們開源了 [DiffSynth-ComfyUI](https://github.com/modelscope/DiffSynth-ComfyUI)，現在，你可以在 [ComfyUI](https://github.com/Comfy-Org/ComfyUI) 中使用 DiffSynth-Studio 執行模型推理了。

- **2026年9月8日** 我們為 CFG 蒸餾的 MiniMax-H3 基座訓練了 [MiniMax-H3 Training Adapter](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter)，這是一個 DeCFG LoRA 訓練 Adapter，提供 FL2VA 與 Ref2VA 兩個版本。訓練資料來自自生成資料集 [MiniMax-H3-Self-Generated-Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/MiniMax-H3-Self-Generated-Dataset)。基於該 Adapter，我們訓練了兩個 toy 模型，[MiniMax-H3-Songyu-LoRA](https://www.modelscope.cn/models/mibei0804/MiniMax-H3-Songyu-LoRA)（角色身份，FL2VA）與 [MiniMax-H3-Ref2VA-FirstFrame-Lineart](https://www.modelscope.cn/models/mibei0804/MiniMax-H3-Ref2VA-FirstFrame-Lineart)（線稿首幀控制，Ref2VA），供推理與微調參考。

- **2026年9月1日** 我們接入了 [SenseNova-U1.5](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT)，這是商湯科技開源的統一多模態模型，我們為其提供了文生圖、影像編輯、低視訊記憶體推理和訓練支援。詳情請參考[文件](/docs/zh/Model_Details/SenseNova-U1.md)和[示例程式碼](/examples/sensenova_u1/)。

- **2026年8月31日** 我們接入了 [Qwen-Video-Edit](https://modelscope.cn/models/yunpeng1998/Qwen-Video-Edit)，這是開源社群使用者 [yunpeng1998](https://github.com/yunpeng1998) 基於影像編輯模型 Qwen-Image-Edit 訓練的影片編輯模型，是探索模型能力拓展的優秀案例。

- **2026年8月25日** 我們開源了 [DiffSynth-WebUI](https://github.com/modelscope/DiffSynth-WebUI)，一鍵私有化部署 LoRA 訓練服務，與模型引數量化功能結合，用消費級 GPU 也能訓練超大模型。

- **2026年8月19日** 我們釋出了模型量化功能。它提供了統一的 `QuantizeConfig` 入口，支援 bitsandbytes、torchao、comfy-kitchen 等多種量化後端，具備線上量化、載入預量化權重、混合量化、儲存量化模型、量化 + LoRA 訓練等能力。詳情請參考[文件](/docs/zh/Pipeline_Usage/Quantization.md)。

<details>
<summary>更多</summary>

- **2026年8月17日** MiniMax-Music3 開源，歡迎加入音訊生成模型家族！支援文生音樂推理和低視訊記憶體推理能力。詳情請參考[文件](/docs/zh/Model_Details/MiniMax-Music3.md)和[示例程式碼](/examples/minimax_music3/)。

- **2026年8月7日** 我們為 Wan 系列新增了 Wan-Animate-2，輸入一張參考圖和一段驅動影片，即可讓參考角色演繹驅動影片中的動作，生成高品質角色動畫，包含標準與蒸餾兩個變體。詳情請參考[文件](/docs/zh/Model_Details/Wan.md)和[示例程式碼](/examples/wanvideo/)。

- **2026年8月3日** MiniMax-H3 開源，歡迎加入影片生成模型家族！支援文生音影片、首尾幀引導生成、參考驅動生成、低視訊記憶體推理以及 NF4 量化推理。詳情請參考[文件](/docs/zh/Model_Details/MiniMax-H3.md)和[示例程式碼](/examples/minimax_h3/)。

- **2026年7月28日** LingBot-Video 開源，歡迎加入影片生成模型家族！本次開源包含 Dense-1.3B 和 MoE-30B-A3B 兩個版本（MoE 版總引數量 30B、每個 token 啟用約 3B），均支援文生影片、圖生影片、文生圖推理、低視訊記憶體推理以及 LoRA / 全量訓練能力。詳情請參考[文件](/docs/zh/Model_Details/LingBot-Video.md)和[示例程式碼](/examples/lingbot_video/)。特別感謝 [NancyFyong](https://github.com/NancyFyong) 貢獻了本模型的接入程式碼！

- **2026年7月21日** 我們開源了 [DiffSynth-Studio Model Integration Skills](https://www.modelscope.cn/collections/DiffSynth-Studio/DiffSynth-Studio-Model-Integration-Skills)。這是一套可組合的 Agent Skill 合集，將外部擴散模型接入 DiffSynth-Studio 的全流程自動化，大幅提升模型接入標準化程度與效率。從[使用示例](https://www.modelscope.cn/skills/DiffSynth-Studio/diffsynth-integrator/file/view/master/example.md?status=1)開始體驗吧！

- **2026年6月29日** Boogu-Image 開源，已支援文生圖推理、影像編輯、低視訊記憶體推理和訓練能力。詳情請參考[文件](/docs/zh/Model_Details/Boogu-Image.md)和[示例程式碼](/examples/boogu_image/)。

- **2026年6月24日** Krea-2 開源，我們已提供全面支援。詳情請參考[文件](/docs/zh/Model_Details/Krea-2.md)和[示例程式碼](/examples/krea2/)。

- **2026年6月16日** 我們為 ACE-Step 新增了 Template 模型：[vocals2music](https://www.modelscope.cn/models/DiffSynth-Studio/acestep15xlsft-vocals2music)。詳情請參考[文件](/docs/zh/Model_Details/ACE-Step.md)和[示例程式碼](/examples/ace_step/)。

- **2026年6月15日** 我們開源了 Image-to-LoRA V2，將動輒數小時的影像風格 LoRA 訓練壓縮到一次模型推理中，探索 LoRA 模型訓練的新方式。[技術報告](https://arxiv.org/abs/2606.13809)已公開，本次開源包括三個模型：
    * [DiffSynth-Studio/ZImage-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/ZImage-i2L-v2)：適配模型 Z-Image
    * [DiffSynth-Studio/KleinBase4B-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/KleinBase4B-i2L-v2)：適配模型 FLUX.2-klein-base-4B
    * [DiffSynth-Studio/HidreamO1-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/HidreamO1-i2L-v2)：適配模型 Hidream-O1-Image

- **2026年6月5日** Ideogram 4 開源，已支援文生圖推理。詳情請參考[文件](/docs/zh/Model_Details/Ideogram-4.md)和[示例程式碼](/examples/ideogram4/)。

- **2026年5月21日** 新增影像品質評估模型的支援，包括 FID、CLIP、Aesthetic、PickScore、ImageReward、HPSv2、HPSv3，詳情請參考[文件](/docs/zh/Model_Details/Image-Quality-Metrics.md)和[示例程式碼](/examples/image_quality_metric/)

- **2026年5月18日** 新增 **CPU Offload Training** 功能，通過將模型權重逐層在 CPU 與 GPU 之間搬運，大幅降低訓練時的 GPU 視訊記憶體佔用，讓消費級顯示卡也能進行大模型 LoRA 訓練，適配所有模型。只需在訓練命令中新增 `--enable_model_cpu_offload` 即可啟用（當前僅支援單卡訓練）。詳情請參考[文件](/docs/zh/Training/Offload_Training.md)。

- **2026年5月14日** HiDream-O1-Image 開源，歡迎加入影像生成模型家族！支援文生圖推理、影像編輯推理、低視訊記憶體推理和訓練能力。詳情請參考[文件](/docs/zh/Model_Details/HiDream-O1-Image.md)和[示例程式碼](/examples/hidream_o1_image/)。

- **2026年4月28日** 我們釋出了 Diffusion Templates，面向 Diffusion 模型的外掛框架，大幅降低了可控生成模型的訓練門檻，一起來探索新奇的技術吧！
    * 開原始碼：[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
    * 技術報告：[arXiv](https://arxiv.org/abs/2604.24351)
    * 專案主頁：[GitHub](https://modelscope.github.io/diffusion-templates-web/)
    * 文件參考：[English Version](https://diffsynth-studio-doc.readthedocs.io/en/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)、[中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)
    * 線上體驗：[魔搭社群創空間](https://modelscope.cn/studios/DiffSynth-Studio/Diffusion-Templates)
    * 模型集：[ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/KleinBase4B-Templates)、[ModelScope 國際站](https://modelscope.ai/collections/DiffSynth-Studio/KleinBase4B-Templates)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/kleinbase4b-templates)
    * 資料集：[ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2)、[ModelScope 國際站](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/imagepulsev2)
- **2026年4月27日** 我們支援了 ACE-Step-1.5！包括文生音樂推理、低視訊記憶體推理和 LoRA 訓練能力。詳情請參考[文件](/docs/zh/Model_Details/ACE-Step.md)和[示例程式碼](/examples/ace_step/)。

- **2026年4月27日** 我們重新支援了模型 Stable Diffusion v1.5 和 SDXL，僅對這兩類模型提供學術科研支援。

- **2026年4月14日** JoyAI-Image 開源，歡迎加入影像編輯模型家族！支援指令引導的影像編輯推理、低視訊記憶體推理和訓練能力。詳情請參考[文件](/docs/zh/Model_Details/JoyAI-Image.md)和[示例程式碼](/examples/joyai_image/)。

- **2026年3月19日** 新增對 [openmoss/MOVA-720p](https://modelscope.cn/models/openmoss/MOVA-720p) 和 [openmoss/MOVA-360p](https://modelscope.cn/models/openmoss/MOVA-360p) 模型的支援，包括完整的訓練和推理功能。[文件](/docs/zh/Model_Details/Wan.md)和[示例程式碼](/examples/mova/)現已可用。

- **2026年3月12日** 我們新增了 [LTX-2.3](https://modelscope.cn/models/Lightricks/LTX-2.3) 音影片生成模型的支援，模型支援的功能包括文生音影片、圖生音影片、IC-LoRA控制、音訊生影片、音影片區域性Inpainting，框架支援完整的推理和訓練功能。詳細資訊請參考 [文件](/docs/zh/Model_Details/LTX-2.md) 和 [示例程式碼](/examples/ltx2/)。

- **2026年3月3日** 我們釋出了 [DiffSynth-Studio/Qwen-Image-Layered-Control-V2](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Layered-Control-V2) 模型，這是 Qwen-Image-Layered-Control 的更新版本。除了原本就支援的文字引導功能，新增了畫筆控制的圖層拆分能力。

- **2026年3月2日** 新增對[Anima](https://modelscope.cn/models/circlestone-labs/Anima)的支援，詳見[文件](docs/zh/Model_Details/Anima.md)。這是一個有趣的動漫風格影像生成模型，我們期待其後續的模型更新。

- **2026年2月26日** 新增對[LTX-2](https://www.modelscope.cn/models/Lightricks/LTX-2)音影片生成模型全量微調與LoRA訓練支援，詳見[文件](docs/zh/Model_Details/LTX-2.md)。

- **2026年2月10日** 新增對[LTX-2](https://www.modelscope.cn/models/Lightricks/LTX-2)音影片生成模型的推理支援，詳見[文件](docs/zh/Model_Details/LTX-2.md)，後續將推進模型訓練的支援。

- **2026年2月2日** Research Tutorial 的第一篇文件上線，帶你從零開始訓練一個 0.1B 的小型文生圖模型，詳見[文件](/docs/zh/Research_Tutorial/train_from_scratch.md)、[模型](https://modelscope.cn/models/DiffSynth-Studio/AAAMyModel)，我們希望 DiffSynth-Studio 能夠成為一個更強大的 Diffusion 模型訓練框架。

- **2026年1月27日** [Z-Image](https://modelscope.cn/models/Tongyi-MAI/Z-Image) 釋出，我們的 [Z-Image-i2L](https://www.modelscope.cn/models/DiffSynth-Studio/Z-Image-i2L) 模型同步釋出，在[魔搭創空間](https://modelscope.cn/studios/DiffSynth-Studio/Z-Image-i2L)可直接體驗，詳見[文件](/docs/zh/Model_Details/Z-Image.md)。

- **2026年1月19日** 新增對 [FLUX.2-klein-4B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B) 和 [FLUX.2-klein-9B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-9B) 模型的支援，包括完整的訓練和推理功能。[文件](/docs/zh/Model_Details/FLUX2.md)和[示例程式碼](/examples/flux2/)現已可用。

- **2026年1月12日** 我們訓練並開源了一個文字引導的圖層拆分模型（[模型連結](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Layered-Control)），這一模型輸入一張圖與一段文字描述，模型會將影像中與文字描述相關的圖層拆分出來。更多細節請閱讀我們的 blog（[中文版](https://modelscope.cn/learn/4938)、[英文版](https://huggingface.co/blog/kelseye/qwen-image-layered-control)）。

- **2025年12月24日** 我們基於 Qwen-Image-Edit-2511 訓練了一個 In-Context Editing LoRA 模型（[模型連結](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-2511-ICEdit-LoRA)），這個模型可以輸入三張圖：圖A、圖B、圖C，模型會自行分析圖A到圖B的變化，並將這樣的變化應用到圖C，生成圖D。更多細節請閱讀我們的 blog（[中文版](https://mp.weixin.qq.com/s/41aEiN3lXKGCJs1-we4Q2g)、[英文版](https://huggingface.co/blog/kelseye/qwen-image-edit-2511-icedit-lora)）。

- **2025年12月9日** 我們基於 DiffSynth-Studio 2.0 訓練了一個瘋狂的模型：[Qwen-Image-i2L](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-i2L)（Image to LoRA）。這一模型以影像為輸入，以 LoRA 為輸出。儘管這個版本的模型在泛化能力、細節保持能力等方面還有很大改進空間，我們將這些模型開源，以啟發更多創新性的研究工作。更多細節，請參考我們的 [blog](https://huggingface.co/blog/kelseye/qwen-image-i2l)。

- **2025年12月4日** DiffSynth-Studio 2.0 釋出！眾多新功能上線
  - [文件](/docs/zh/README.md)上線：我們的文件還在持續最佳化更新中
  - [視訊記憶體管理](/docs/zh/Pipeline_Usage/VRAM_management.md)模組升級，支援 Layer 級別的 Disk Offload，同時釋放記憶體與視訊記憶體
  - 新模型支援
    - Z-Image Turbo: [模型](https://www.modelscope.ai/models/Tongyi-MAI/Z-Image-Turbo)、[文件](/docs/zh/Model_Details/Z-Image.md)、[程式碼](/examples/z_image/)
    - FLUX.2-dev: [模型](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-dev)、[文件](/docs/zh/Model_Details/FLUX2.md)、[程式碼](/examples/flux2/)
  - 訓練框架升級
    - [拆分訓練](/docs/zh/Training/Split_Training.md)：支援自動化地將訓練過程拆分為資料處理和訓練兩階段（即使訓練的是 ControlNet 或其他任意模型），在資料處理階段進行文字編碼、VAE 編碼等不需要梯度回傳的計算，在訓練階段處理其他計算。速度更快，視訊記憶體需求更少。
    - [差分 LoRA 訓練](/docs/zh/Training/Differential_LoRA.md)：這是我們曾在 [ArtAug](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1) 中使用的訓練技術，目前已可用於任意模型的 LoRA 訓練。
    - [FP8 訓練](/docs/zh/Training/FP8_Precision.md)：FP8 在訓練中支援應用到任意非訓練模型，即梯度關閉或者梯度僅影響 LoRA 權重的模型。
- **2025年11月4日** 支援了 [ByteDance/Video-As-Prompt-Wan2.1-14B](https://modelscope.cn/models/ByteDance/Video-As-Prompt-Wan2.1-14B) 模型，該模型基於 Wan 2.1 訓練，支援根據參考影片生成相應的動作。

- **2025年10月30日** 支援了 [meituan-longcat/LongCat-Video](https://www.modelscope.cn/models/meituan-longcat/LongCat-Video) 模型，該模型支援文生影片、圖生影片、影片續寫。這個模型在本專案中沿用 Wan 的框架進行推理和訓練。

- **2025年10月27日** 支援了 [krea/krea-realtime-video](https://www.modelscope.cn/models/krea/krea-realtime-video) 模型，Wan 模型生態再添一員。

- **2025年9月23日** [DiffSynth-Studio/Qwen-Image-EliGen-Poster](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-Poster) 釋出！本模型由我們與淘天體驗設計團隊聯合研發並開源。模型基於 Qwen-Image 構建，專為電商海報場景設計，支援精確的分割槽佈局控制。 請參考[我們的示例程式碼](./examples/qwen_image/model_inference/Qwen-Image-EliGen-Poster.py)。

- **2025年9月9日** 我們的訓練框架支援了多種訓練模式，目前已適配 Qwen-Image，除標準 SFT 訓練模式外，已支援 Direct Distill，請參考[我們的示例程式碼](./examples/qwen_image/model_training/lora/Qwen-Image-Distill-LoRA.sh)。這項功能是實驗性的，我們將會繼續完善已支援更全面的模型訓練功能。

- **2025年8月28日** 我們支援了Wan2.2-S2V，一個音訊驅動的電影級影片生成模型。請參見[./examples/wanvideo/](./examples/wanvideo/)。

- **2025年8月21日** [DiffSynth-Studio/Qwen-Image-EliGen-V2](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-V2) 釋出！相比於 V1 版本，訓練資料集變為 [Qwen-Image-Self-Generated-Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Qwen-Image-Self-Generated-Dataset)，因此，生成的影像更符合 Qwen-Image 本身的影像分佈和風格。 請參考[我們的示例程式碼](./examples/qwen_image/model_inference_low_vram/Qwen-Image-EliGen-V2.py)。

- **2025年8月21日** 我們開源了 [DiffSynth-Studio/Qwen-Image-In-Context-Control-Union](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-In-Context-Control-Union) 結構控制 LoRA 模型，採用 In Context 的技術路線，支援多種類別的結構控制條件，包括 canny, depth, lineart, softedge, normal, openpose。 請參考[我們的示例程式碼](./examples/qwen_image/model_inference/Qwen-Image-In-Context-Control-Union.py)。

- **2025年8月20日** 我們開源了 [DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix) 模型，提升了 Qwen-Image-Edit 對低解析度影像輸入的編輯效果。請參考[我們的示例程式碼](./examples/qwen_image/model_inference/Qwen-Image-Edit-Lowres-Fix.py)

- **2025年8月19日** Qwen-Image-Edit 開源，歡迎影像編輯模型新成員！

- **2025年8月18日** 我們訓練並開源了 Qwen-Image 的影像重繪 ControlNet 模型 [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint)，模型結構採用了輕量化的設計，請參考[我們的示例程式碼](./examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Inpaint.py)。

- **2025年8月15日** 我們開源了 [Qwen-Image-Self-Generated-Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Qwen-Image-Self-Generated-Dataset) 資料集。這是一個使用 Qwen-Image 模型生成的影像資料集，共包含 160,000 張`1024 x 1024`影像。它包括通用、英文文字渲染和中文文字渲染子集。我們為每張影像提供了影像描述、實體和結構控制影像的標註。開發者可以使用這個資料集來訓練 Qwen-Image 模型的 ControlNet 和 EliGen 等模型，我們旨在通過開源推動技術發展！

- **2025年8月13日** 我們訓練並開源了 Qwen-Image 的 ControlNet 模型 [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth)，模型結構採用了輕量化的設計，請參考[我們的示例程式碼](./examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Depth.py)。

- **2025年8月12日** 我們訓練並開源了 Qwen-Image 的 ControlNet 模型 [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny)，模型結構採用了輕量化的設計，請參考[我們的示例程式碼](./examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Canny.py)。

- **2025年8月11日** 我們開源了 Qwen-Image 的蒸餾加速模型 [DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA)，沿用了與 [DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full) 相同的訓練流程，但模型結構修改為了 LoRA，因此能夠更好地與其他開源生態模型相容。

- **2025年8月7日** 我們開源了 Qwen-Image 的實體控制 LoRA 模型 [DiffSynth-Studio/Qwen-Image-EliGen](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen)。Qwen-Image-EliGen 能夠實現實體級可控的文生圖。技術細節請參見[論文](https://arxiv.org/abs/2501.01097)。訓練資料集：[EliGenTrainSet](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet)。

- **2025年8月5日** 我們開源了 Qwen-Image 的蒸餾加速模型 [DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full)，實現了約 5 倍加速。

- **2025年8月4日** Qwen-Image 開源，歡迎影像生成模型家族新成員！

- **2025年8月1日** [FLUX.1-Krea-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Krea-dev) 開源，這是一個專注於美學攝影的文生圖模型。我們第一時間提供了全方位支援，包括低視訊記憶體逐層 offload、LoRA 訓練、全量訓練。詳細資訊請參考 [./examples/flux/](./examples/flux/)。

- **2025年7月28日** Wan 2.2 開源，我們第一時間提供了全方位支援，包括低視訊記憶體逐層 offload、FP8 量化、序列並行、LoRA 訓練、全量訓練。詳細資訊請參考 [./examples/wanvideo/](./examples/wanvideo/)。

- **2025年7月11日** 我們提出 Nexus-Gen，一個將大語言模型（LLM）的語言推理能力與擴散模型的影像生成能力相結合的統一框架。該框架支援無縫的影像理解、生成和編輯任務。
  - 論文: [Nexus-Gen: Unified Image Understanding, Generation, and Editing via Prefilled Autoregression in Shared Embedding Space](https://arxiv.org/pdf/2504.21356)
  - Github 倉庫: https://github.com/modelscope/Nexus-Gen
  - 模型: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2), [HuggingFace](https://huggingface.co/modelscope/Nexus-GenV2)
  - 訓練資料集: [ModelScope Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Nexus-Gen-Training-Dataset)
  - 線上體驗: [ModelScope Nexus-Gen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/Nexus-Gen)
- **2025年6月15日** ModelScope 官方評測框架 [EvalScope](https://github.com/modelscope/evalscope) 現已支援文生圖生成評測。請參考[最佳實踐](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/t2i_eval.html)指南進行嘗試。

- **2025年3月25日** 我們的新開源專案 [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) 現已開源！專注於穩定的模型部署，面向工業界，提供更好的工程支援、更高的計算效能和更穩定的功能。

- **2025年3月31日** 我們支援 InfiniteYou，一種用於 FLUX 的人臉特徵保留方法。更多細節請參考 [./examples/InfiniteYou/](./examples/InfiniteYou/)。

- **2025年3月13日** 我們支援 HunyuanVideo-I2V，即騰訊開源的 HunyuanVideo 的影像到影片生成版本。更多細節請參考 [./examples/HunyuanVideo/](./examples/HunyuanVideo/)。

- **2025年2月25日** 我們支援 Wan-Video，這是阿里巴巴開源的一系列最先進的影片合成模型。詳見 [./examples/wanvideo/](./examples/wanvideo/)。

- **2025年2月17日** 我們支援 [StepVideo](https://modelscope.cn/models/stepfun-ai/stepvideo-t2v/summary)！先進的影片合成模型！詳見 [./examples/stepvideo](./examples/stepvideo/)。

- **2024年12月31日** 我們提出 EliGen，一種用於精確實體級別控制的文字到影像生成的新框架，並輔以修復融合管道，將其能力擴充套件到影像修復任務。EliGen 可以無縫整合現有的社群模型，如 IP-Adapter 和 In-Context LoRA，提升其通用性。更多詳情，請見 [./examples/EntityControl](./examples/EntityControl/)。
  - 論文: [EliGen: Entity-Level Controlled Image Generation with Regional Attention](https://arxiv.org/abs/2501.01097)
  - 模型: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen), [HuggingFace](https://huggingface.co/modelscope/EliGen)
  - 線上體驗: [ModelScope EliGen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/EliGen)
  - 訓練資料集: [EliGen Train Set](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet)
- **2024年12月19日** 我們為 HunyuanVideo 實現了進階視訊記憶體管理，使得在 24GB 視訊記憶體下可以生成解析度為 129x720x1280 的影片，或在僅 6GB 視訊記憶體下生成解析度為 129x512x384 的影片。更多細節請參考 [./examples/HunyuanVideo/](./examples/HunyuanVideo/)。

- **2024年12月18日** 我們提出 ArtAug，一種通過合成-理解互動來改進文生圖模型的方法。我們以 LoRA 格式為 FLUX.1-dev 訓練了一個 ArtAug 增強模組。該模型將 Qwen2-VL-72B 的美學理解融入 FLUX.1-dev，從而提升了生成影像的品質。
  - 論文: https://arxiv.org/abs/2412.12888
  - 示例: https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/ArtAug
  - 模型: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ArtAug-lora-FLUX.1dev-v1)
  - 演示: [ModelScope](https://modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=7228&modelType=LoRA&sdVersion=FLUX_1&modelUrl=modelscope%3A%2F%2FDiffSynth-Studio%2FArtAug-lora-FLUX.1dev-v1%3Frevision%3Dv1.0), HuggingFace (即將上線)
- **2024年10月25日** 我們提供了廣泛的 FLUX ControlNet 支援。該專案支援許多不同的 ControlNet 模型，並且可以自由組合，即使它們的結構不同。此外，ControlNet 模型相容高解析度最佳化和分割槽控制技術，能夠實現非常強大的可控影像生成。詳見 [`./examples/ControlNet/`](./examples/ControlNet/)。

- **2024年10月8日** 我們釋出了基於 CogVideoX-5B 和 ExVideo 的擴充套件 LoRA。您可以從 [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1) 或 [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1) 下載此模型。

- **2024年8月22日** 本專案現已支援 CogVideoX-5B。詳見 [此處](/examples/video_synthesis/)。我們為這個文生影片模型提供了幾個有趣的功能，包括：
  - 文字到影片
  - 影片編輯
  - 自我超分
  - 影片插幀
- **2024年8月22日** 我們實現了一個有趣的畫筆功能，支援所有文生圖模型。現在，您可以在 AI 的輔助下使用畫筆創作驚豔的影像了！
  - 在我們的 [WebUI](#usage-in-webui) 中使用它。
- **2024年8月21日** DiffSynth-Studio 現已支援 FLUX。
  - 啟用 CFG 和高解析度修復以提升視覺品質。詳見 [此處](/examples/image_synthesis/README.md)
  - LoRA、ControlNet 和其他附加模型將很快推出。
- **2024年6月21日** 我們提出 ExVideo，一種旨在增強影片生成模型能力的後訓練微調技術。我們將 Stable Video Diffusion 進行了擴充套件，實現了長達 128 幀的長影片生成。
  - [專案頁面](https://ecnu-cilab.github.io/ExVideoProjectPage/)
  - 原始碼已在此倉庫中釋出。詳見 [`examples/ExVideo`](./examples/ExVideo/)。
  - 模型已釋出於 [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1) 和 [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1)。
  - 技術報告已釋出於 [arXiv](https://arxiv.org/abs/2406.14130)。
  - 您可以在此 [演示](https://huggingface.co/spaces/modelscope/ExVideo-SVD-128f-v1) 中試用 ExVideo！
- **2024年6月13日** DiffSynth Studio 已遷移至 ModelScope。開發團隊也從“我”轉變為“我們”。當然，我仍會參與後續的開發和維護工作。

- **2024年1月29日** 我們提出 Diffutoon，這是一個出色的卡通著色解決方案。
  - [專案頁面](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
  - 原始碼已在此專案中釋出。
  - 技術報告（IJCAI 2024）已釋出於 [arXiv](https://arxiv.org/abs/2401.16224)。
- **2023年12月8日** 我們決定啟動一個新專案，旨在釋放擴散模型的潛力，尤其是在影片合成方面。該專案的開發工作正式開始。

- **2023年11月15日** 我們提出 FastBlend，一種強大的影片去閃爍演算法。
  - sd-webui 擴充套件已釋出於 [GitHub](https://github.com/Artiprocher/sd-webui-fastblend)。
  - 演示影片已在 Bilibili 上展示，包含三個任務：
    - [影片去閃爍](https://www.bilibili.com/video/BV1d94y1W7PE)
    - [影片插幀](https://www.bilibili.com/video/BV1Lw411m71p)
    - [影像驅動的影片渲染](https://www.bilibili.com/video/BV1RB4y1Z7LF)
  - 技術報告已釋出於 [arXiv](https://arxiv.org/abs/2311.09265)。
  - 其他使用者開發的非官方 ComfyUI 擴充套件已釋出於 [GitHub](https://github.com/AInseven/ComfyUI-fastblend)。
- **2023年10月1日** 我們釋出了該專案的早期版本，名為 FastSDXL。這是構建一個擴散引擎的初步嘗試。
  - 原始碼已釋出於 [GitHub](https://github.com/Artiprocher/FastSDXL)。
  - FastSDXL 包含一個可訓練的 OLSS 排程器，以提高效率。
    - OLSS 的原始倉庫位於 [此處](https://github.com/alibaba/EasyNLP/tree/master/diffusion/olss_scheduler)。
    - 技術報告（CIKM 2023）已釋出於 [arXiv](https://arxiv.org/abs/2305.14677)。
    - 演示影片已釋出於 [Bilibili](https://www.bilibili.com/video/BV1w8411y7uj)。
    - 由於 OLSS 需要額外訓練，我們未在本專案中實現它。
- **2023年8月29日** 我們提出 DiffSynth，一個影片合成框架。
  - [專案頁面](https://ecnu-cilab.github.io/DiffSynth.github.io/)。
  - 原始碼已釋出在 [EasyNLP](https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth)。
  - 技術報告（ECML PKDD 2024）已釋出於 [arXiv](https://arxiv.org/abs/2308.03463)。

</details>

## 安裝

從原始碼安裝（推薦）：

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多安裝方式，以及非 NVIDIA GPU 的安裝，請參考[安裝文件](/docs/zh/Pipeline_Usage/Setup.md)。

<details>
<summary>下載源配置</summary>

> 在進行模型推理和訓練前，可通過[環境變數](/docs/zh/Pipeline_Usage/Environment_Variables.md)配置模型下載源等。
> 
> 本專案預設從[魔搭社群](https://modelscope.cn/)下載模型。對於非中國區域的使用者，可以通過以下配置從[魔搭社群國際站](https://modelscope.ai)下載模型：
> 
> ```shell
> export MODELSCOPE_ENDPOINT=https://modelscope.ai
> ```
> 
> 如需從 [HuggingFace](https://huggingface.co/) 下載模型，請修改[環境變數](/docs/zh/Pipeline_Usage/Environment_Variables.md)，注意不同模型平臺上的模型 ID 可能不同：
> 
> ```shell
> export DIFFSYNTH_DOWNLOAD_SOURCE="huggingface"
> ```

</details>

## 基礎框架

DiffSynth-Studio 作為基礎的模型框架，為主流 Diffusion 模型重新設計了推理和訓練流水線，能夠實現高效的視訊記憶體管理、靈活的模型訓練。

快速開始，體驗熱門和最新模型：

| 基礎架構 | 模型 ID | 推理 | 低視訊記憶體推理 | 全量訓練 | 全量訓練後驗證 | LoRA 訓練 | LoRA 訓練後驗證 |
|-|-|-|-|-|-|-|-|
| Fibo 2.0 | [briaai/Fibo-2.0-dev](https://www.modelscope.cn/models/briaai/Fibo-2.0-dev) | [code](/examples/fibo/model_inference/Fibo-2.0-dev.py) | [code](/examples/fibo/model_inference_low_vram/Fibo-2.0-dev.py) | [code](/examples/fibo/model_training/full/Fibo-2.0-dev-T2I.sh) | [code](/examples/fibo/model_training/validate_full/Fibo-2.0-dev-T2I.py) | [code](/examples/fibo/model_training/lora/Fibo-2.0-dev-T2I.sh) | [code](/examples/fibo/model_training/validate_lora/Fibo-2.0-dev-T2I.py) |
| Fibo 2.0 | [briaai/Fibo-2.0-edit-turbo](https://www.modelscope.cn/models/briaai/Fibo-2.0-edit-turbo) | [code](/examples/fibo/model_inference/Fibo-2.0-edit-turbo.py) | [code](/examples/fibo/model_inference_low_vram/Fibo-2.0-edit-turbo.py) | - | - | - | - |
| Qwen-Image-2.1 | [Qwen/Qwen-Image-2.1](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1) | [code](/examples/qwen_image_21/model_inference/Qwen-Image-2.1.py) | [code](/examples/qwen_image_21/model_inference_low_vram/Qwen-Image-2.1.py) | [code](/examples/qwen_image_21/model_training/full/Qwen-Image-2.1.sh) | [code](/examples/qwen_image_21/model_training/validate_full/Qwen-Image-2.1.py) | [code](/examples/qwen_image_21/model_training/lora/Qwen-Image-2.1.sh) | [code](/examples/qwen_image_21/model_training/validate_lora/Qwen-Image-2.1.py) |
| Qwen-Image-2.1 | [Qwen/Qwen-Image-2.1-Turbo](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo) | [code](/examples/qwen_image_21/model_inference/Qwen-Image-2.1-Turbo.py) | [code](/examples/qwen_image_21/model_inference_low_vram/Qwen-Image-2.1-Turbo.py) | [code](/examples/qwen_image_21/model_training/full/Qwen-Image-2.1-Turbo.sh) | [code](/examples/qwen_image_21/model_training/validate_full/Qwen-Image-2.1-Turbo.py) | [code](/examples/qwen_image_21/model_training/lora/Qwen-Image-2.1-Turbo.sh) | [code](/examples/qwen_image_21/model_training/validate_lora/Qwen-Image-2.1-Turbo.py) |
| MiniMax-H3 | [MiniMax/MiniMax-H3: FL2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FL2VA.py) |
| MiniMax-H3 | [DiffSynth-Studio/MiniMax-H3-NF4: FL2VA pruned](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-NF4-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-Pruned-FL2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Pruned-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-Pruned-FL2VA.py) |
| ACE-Step | [ACE-Step/acestep-v15-xl-sft](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-sft) | [code](/examples/ace_step/model_inference/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_training/full/acestep-v15-xl-sft.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-xl-sft.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-xl-sft.py) |
| Z-Image | [Tongyi-MAI/Z-Image-Turbo](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) | [code](/examples/z_image/model_inference/Z-Image-Turbo.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image-Turbo.py) | [code](/examples/z_image/model_training/full/Z-Image-Turbo.sh) | [code](/examples/z_image/model_training/validate_full/Z-Image-Turbo.py) | [code](/examples/z_image/model_training/lora/Z-Image-Turbo.sh) | [code](/examples/z_image/model_training/validate_lora/Z-Image-Turbo.py) |
| Krea-2 | [krea/Krea-2-Raw](https://www.modelscope.cn/models/krea/Krea-2-Raw) | [code](/examples/krea2/model_inference/Krea-2-Raw.py) | [code](/examples/krea2/model_inference_low_vram/Krea-2-Raw.py) | [code](/examples/krea2/model_training/full/Krea-2-Raw.sh) | [code](/examples/krea2/model_training/validate_full/Krea-2-Raw.py) | [code](/examples/krea2/model_training/lora/Krea-2-Raw.sh) | [code](/examples/krea2/model_training/validate_lora/Krea-2-Raw.py) |
| Krea-2 | [krea/Krea-2-Turbo](https://www.modelscope.cn/models/krea/Krea-2-Turbo) | [code](/examples/krea2/model_inference/Krea-2-Turbo.py) | [code](/examples/krea2/model_inference_low_vram/Krea-2-Turbo.py) | [code](/examples/krea2/model_training/full/Krea-2-Turbo.sh) | [code](/examples/krea2/model_training/validate_full/Krea-2-Turbo.py) | [code](/examples/krea2/model_training/lora/Krea-2-Turbo.sh) | [code](/examples/krea2/model_training/validate_lora/Krea-2-Turbo.py) |

模型一覽：

- 影像生成
    - Fibo 2.0：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Fibo.html)、[樣例程式碼](/examples/fibo/)
    - MOSS-Image：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/MOSS-Image.html)、[樣例程式碼](/examples/moss_image/)
    - FLUX.3：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/FLUX.3.html)、[樣例程式碼](/examples/flux3/)
    - Qwen-Image-2.1：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Qwen-Image-2.1.html)、[樣例程式碼](/examples/qwen_image_21/)
    - SenseNova-U1：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/SenseNova-U1.html)、[樣例程式碼](/examples/sensenova_u1/)
    - Boogu-Image：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Boogu-Image.html)、[樣例程式碼](/examples/boogu_image/)
    - Krea-2：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Krea-2.html)、[樣例程式碼](/examples/krea2/)
    - Ideogram 4：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Ideogram-4.html)、[樣例程式碼](/examples/ideogram4/)
    - HiDream-O1-Image：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/HiDream-O1-Image.html)、[樣例程式碼](/examples/hidream_o1_image/)
    - JoyAI-Image：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/JoyAI-Image.html)、[樣例程式碼](/examples/joyai_image/)
    - ERNIE-Image：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/ERNIE-Image.html)、[樣例程式碼](/examples/ernie_image/)
    - FLUX.2：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/FLUX2.html)、[樣例程式碼](/examples/flux2/)
    - Z-Image：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Z-Image.html)、[樣例程式碼](/examples/z_image/)
    - Anima：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Anima.html)、[樣例程式碼](/examples/anima/)
    - Qwen-Image：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Qwen-Image.html)、[樣例程式碼](/examples/qwen_image/)
    - FLUX.1：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/FLUX.html)、[樣例程式碼](/examples/flux/)
    - Stable Diffusion XL：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Stable-Diffusion-XL.html)、[樣例程式碼](/examples/stable_diffusion_xl/)
    - Stable Diffusion：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Stable-Diffusion.html)、[樣例程式碼](/examples/stable_diffusion/)
- 影片生成
    - MiniMax-H3：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/MiniMax-H3.html)、[樣例程式碼](/examples/minimax_h3/)
    - LingBot-Video：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/LingBot-Video.html)、[樣例程式碼](/examples/lingbot_video/)
    - LTX-2：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/LTX-2.html)、[樣例程式碼](/examples/ltx2/)
    - Wan：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Wan.html)、[樣例程式碼](/examples/wanvideo/)
- 音訊生成
    - DiffSynth-Music：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/DiffSynth-Music.html)、[樣例程式碼](/examples/diffsynth_music/)
    - MiniMax-Music3：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/MiniMax-Music3.html)、[樣例程式碼](/examples/minimax_music3/)
    - YuE2：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/YuE2.html)、[樣例程式碼](/examples/yue2/)
    - ACE-Step：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/ACE-Step.html)、[樣例程式碼](/examples/ace_step/)
- 評測模型：[文件](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Image-Quality-Metrics.html)、[樣例程式碼](/examples/image_quality_metric/)

[檢視全部支援的模型](#全部支援的模型)

## 創新成果

我們相信，一個完善的開原始碼框架能夠降低技術探索的門檻，我們基於這個程式碼庫搞出了不少有意思的技術。或許你也有許多天馬行空的構想，藉助 DiffSynth-Studio，你可以快速實現這些想法。

<details>
<summary>EntroPack: 面向 PyTorch 的通用張量壓縮庫</summary>

> 按任意目標位元速率壓縮模型權重，GPU 編解碼，壓縮後的權重可直接用於推理與 LoRA 訓練。

* 專案主頁：[GitHub](https://github.com/modelscope/entropack)
* 技術報告：https://arxiv.org/abs/2609.34185
* 預量化模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/EntroPackPreQuants)
* 文件參考：[English Version](/docs/en/Pipeline_Usage/Quantization.md)、[中文版](/docs/zh/Pipeline_Usage/Quantization.md)
* 示例程式碼：[Z-Image-Turbo](/examples/z_image/model_inference_low_vram/Z-Image-Turbo-EntroPack.py)、[Qwen-Image-2.1](/examples/qwen_image_21/model_inference_low_vram/Qwen-Image-2.1-EntroPack.py)、[MiniMax-H3](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-EntroPack.py)、[LoRA 訓練示例](/examples/z_image/model_training/special/quant_training/Z-Image-Turbo-entropack_lossy_quant.sh)

</details>

<details>
<summary>TreeAdapter: 由結構化 LoRA 組建的模型系統</summary>

> 用 1w+ LoRA 組成的模型系統挑戰上萬個稀有物種的精準生成。

* 論文：[TreeAdapter: Hierarchical Taxonomy-Guided Adapter Composition for Fine-Grained Species Image Generation](https://arxiv.org/abs/2607.24215)
* 模型：[ModelScope](https://modelscope.cn/models/DiffSynth-Studio/TreeAdapter-KleinBase4B)

![Image](https://github.com/user-attachments/assets/1b461e0f-60aa-4b38-a44d-d1646cbbbc75)

</details>

<details>
<summary>Image-to-LoRA: 把模型訓練壓縮到模型推理中</summary>

> 真正的 Meta Learning：模型一端輸入影像資料集，另一端直接輸出訓出來的 LoRA 模型。

* 論文：[Compressing Image Style Training into a Single Model Forward](https://arxiv.org/abs/2606.13809)
* 模型：
    * [DiffSynth-Studio/ZImage-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/ZImage-i2L-v2)：適配模型 Z-Image
    * [DiffSynth-Studio/KleinBase4B-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/KleinBase4B-i2L-v2)：適配模型 FLUX.2-klein-base-4B
    * [DiffSynth-Studio/HidreamO1-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/HidreamO1-i2L-v2)：適配模型 Hidream-O1-Image

|輸入樣例1|輸出樣例1|輸入樣例2|輸出樣例2|
|-|-|-|-|
|![Image](https://github.com/user-attachments/assets/0a1fd252-851f-414e-af24-4c656ab54277)|![Image](https://github.com/user-attachments/assets/96259993-e732-424d-bf07-9ca1ede27890)|![Image](https://github.com/user-attachments/assets/a78573a0-c2cf-4e33-ac21-276078e8cad3)|![Image](https://github.com/user-attachments/assets/8177e883-cfef-4e38-a528-cdef01a9f9b8)|

</details>

<details>
<summary>Diffusion-Templates: 外掛化的可控生成框架</summary>

> 一個框架，讓每一種可控生成能力都成為外掛，讓多個模型組合湧現出豐富生成能力。

* 論文：[Diffusion Templates: A Unified Plugin Framework for Controllable Diffusion](https://arxiv.org/abs/2604.24351)
* 專案主頁：[GitHub](https://modelscope.github.io/diffusion-templates-web/)
* 文件參考：[English Version](https://diffsynth-studio-doc.readthedocs.io/en/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)、[中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)
* 線上體驗：[魔搭社群創空間](https://modelscope.cn/studios/DiffSynth-Studio/Diffusion-Templates)
* 模型集：[ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/KleinBase4B-Templates)、[ModelScope 國際站](https://modelscope.ai/collections/DiffSynth-Studio/KleinBase4B-Templates)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/kleinbase4b-templates)
* 資料集：[ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2)、[ModelScope 國際站](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/imagepulsev2)

|參考圖|區域性編輯|風格遷移|清晰度增強|
|-|-|-|-|
|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_reference.jpg)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_Brightness_Edit_Inpaint.png)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_Controlnet_Edit_SoftRGB.png)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_Upscaler_Sharpness.png)|

</details>

<details>
<summary>SES: 用於獎勵對齊影像生成的高效推理階段縮放</summary>

> 以推理時間為代價，提高生成內容的品質。

- 論文：[Spectral Evolution Search: Efficient Inference-Time Scaling for Reward-Aligned Image Generation](https://arxiv.org/abs/2602.03208)
- 程式碼樣例：[/docs/en/Research_Tutorial/inference_time_scaling.md](/docs/en/Research_Tutorial/inference_time_scaling.md)

|FLUX.1-dev|FLUX.1-dev + SES|Qwen-Image|Qwen-Image + SES|
|-|-|-|-|
|![Image](https://github.com/user-attachments/assets/5be15dc6-2805-4822-b04c-2573fc0f45f0)|![Image](https://github.com/user-attachments/assets/e71b8c20-1629-41d9-b0ff-185805c1da4e)|![Image](https://github.com/user-attachments/assets/7a73c968-133a-4545-9aa2-205533861cd4)|![Image](https://github.com/user-attachments/assets/c8390b22-14fe-48a0-a6e6-d6556d31235e)|

</details>

<details>
<summary>VIRAL：基於DiT模型的類比視覺上下文推理</summary>

> 根據圖1到圖2的變化把圖3轉化為圖4，影像編輯模型的能力湧現。

- 論文：[VIRAL: Visual In-Context Reasoning via Analogy in Diffusion Transformers
](https://arxiv.org/abs/2602.03210)
- 程式碼樣例：[/examples/qwen_image/model_inference/Qwen-Image-Edit-2511-ICEdit.py](/examples/qwen_image/model_inference/Qwen-Image-Edit-2511-ICEdit.py)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-2511-ICEdit-LoRA)

|Example 1|Example 2|Query|Output|
|-|-|-|-|
|![Image](https://github.com/user-attachments/assets/380d2670-47bf-41cd-b5c9-37110cc4a943)|![Image](https://github.com/user-attachments/assets/7ceaf345-0992-46e6-b38f-394c2065b165)|![Image](https://github.com/user-attachments/assets/f7c26c21-6894-4d9e-b570-f1d44ca7c1de)|![Image](https://github.com/user-attachments/assets/c2bebe3b-5984-41ba-94bf-9509f6a8a990)|

</details>

<details>
<summary>AttriCtrl: 影像生成模型的屬性強度控制</summary>

> 用數值屬性也能精準地控制影像生成模型。

- 論文：[AttriCtrl: Fine-Grained Control of Aesthetic Attribute Intensity in Diffusion Models
](https://arxiv.org/abs/2508.02151)
- 程式碼樣例：[/examples/flux/model_inference/FLUX.1-dev-AttriCtrl.py](/examples/flux/model_inference/FLUX.1-dev-AttriCtrl.py)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/AttriCtrl-FLUX.1-Dev)

|brightness scale = 0.1|brightness scale = 0.3|brightness scale = 0.5|brightness scale = 0.7|brightness scale = 0.9|
|-|-|-|-|-|
|![Image](https://github.com/user-attachments/assets/e74b32a5-5b2e-4c87-9df8-487c0f8366b7)|![Image](https://github.com/user-attachments/assets/bfe8bec2-9e55-493d-9a26-7e9cce28e03d)|![Image](https://github.com/user-attachments/assets/b099dfe3-ff1f-4b96-894c-d48bbe92db7a)|![Image](https://github.com/user-attachments/assets/0a6b2982-deab-4b0d-91ad-888782de01c9)|![Image](https://github.com/user-attachments/assets/fcecb755-7d03-4020-b83a-13ad2b38705c)|

</details>

<details>
<summary>AutoLoRA: 自動化的 LoRA 檢索和融合</summary>

> LoRA 是需求和解決方案一體化的產物，如何更好地利用這些 LoRA？

- 論文：[AutoLoRA: Automatic LoRA Retrieval and Fine-Grained Gated Fusion for Text-to-Image Generation
](https://arxiv.org/abs/2508.02107)
- 程式碼樣例：[/examples/flux/model_inference/FLUX.1-dev-LoRA-Fusion.py](/examples/flux/model_inference/FLUX.1-dev-LoRA-Fusion.py)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/LoRAFusion-preview-FLUX.1-dev)

||[LoRA 1](https://modelscope.cn/models/cancel13/cxsk)|[LoRA 2](https://modelscope.cn/models/wy413928499/xuancai2)|[LoRA 3](https://modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1)|[LoRA 4](https://modelscope.cn/models/hongyanbujian/JPL)|
|-|-|-|-|-|
|[LoRA 1](https://modelscope.cn/models/cancel13/cxsk)                              |![Image](https://github.com/user-attachments/assets/01c54d5a-4f00-4c2e-982a-4ec0a4c6a6e3)|![Image](https://github.com/user-attachments/assets/e6621457-b9f1-437c-bcc8-3e12e41646de)|![Image](https://github.com/user-attachments/assets/4b7f721f-a2e5-416c-af2c-b53ef236c321)|![Image](https://github.com/user-attachments/assets/802d554e-0402-482c-9f28-87605f8fe318)|
|[LoRA 2](https://modelscope.cn/models/wy413928499/xuancai2)                       |![Image](https://github.com/user-attachments/assets/e6621457-b9f1-437c-bcc8-3e12e41646de)|![Image](https://github.com/user-attachments/assets/43720a9f-aa27-4918-947d-545389375d46)|![Image](https://github.com/user-attachments/assets/418c725b-6d35-41f4-b18f-c7e3867cc142)|![Image](https://github.com/user-attachments/assets/8c8f22fa-9643-4019-b6d7-396d8b7fed9a)|
|[LoRA 3](https://modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1)  |![Image](https://github.com/user-attachments/assets/4b7f721f-a2e5-416c-af2c-b53ef236c321)|![Image](https://github.com/user-attachments/assets/418c725b-6d35-41f4-b18f-c7e3867cc142)|![Image](https://github.com/user-attachments/assets/041a3f9a-c7b4-4311-8582-cb71a7226d80)|![Image](https://github.com/user-attachments/assets/b54ebaa4-31a7-4536-a2c1-496adba0c013)|
|[LoRA 4](https://modelscope.cn/models/hongyanbujian/JPL)                          |![Image](https://github.com/user-attachments/assets/802d554e-0402-482c-9f28-87605f8fe318)|![Image](https://github.com/user-attachments/assets/8c8f22fa-9643-4019-b6d7-396d8b7fed9a)|![Image](https://github.com/user-attachments/assets/b54ebaa4-31a7-4536-a2c1-496adba0c013)|![Image](https://github.com/user-attachments/assets/a640fd54-3192-49a0-9281-b43d9ba64f09)|

</details>

<details>
<summary>Nexus-Gen: 統一架構的影像理解、生成、編輯</summary>

> 如果一個模型集齊影像理解、影像生成、影像編輯能力，會發生什麼？

- 詳細頁面：https://github.com/modelscope/Nexus-Gen
- 論文：[Nexus-Gen: Unified Image Understanding, Generation, and Editing via Prefilled Autoregression in Shared Embedding Space](https://arxiv.org/pdf/2504.21356)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2), [HuggingFace](https://huggingface.co/modelscope/Nexus-GenV2)
- 資料集：[ModelScope Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Nexus-Gen-Training-Dataset)
- 線上體驗：[ModelScope Nexus-Gen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/Nexus-Gen)

![](https://github.com/modelscope/Nexus-Gen/raw/main/assets/illustrations/gen_edit.jpg)

</details>

<details>
<summary>ArtAug: 影像生成模型的美學提升</summary>

> 一個 LoRA，大幅提升細節和美感。

- 詳細頁面：[./examples/ArtAug/](./examples/ArtAug/)
- 論文：[ArtAug: Enhancing Text-to-Image Generation through Synthesis-Understanding Interaction](https://arxiv.org/abs/2412.12888)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ArtAug-lora-FLUX.1dev-v1)
- 線上體驗：[ModelScope AIGC Tab](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=7228&modelType=LoRA&sdVersion=FLUX_1&modelUrl=modelscope%3A%2F%2FDiffSynth-Studio%2FArtAug-lora-FLUX.1dev-v1%3Frevision%3Dv1.0)

|FLUX.1-dev|FLUX.1-dev + ArtAug LoRA|
|-|-|
|![image_1_base](https://github.com/user-attachments/assets/e1d5c505-b423-45fe-be01-25c2758f5417)|![image_1_enhance](https://github.com/user-attachments/assets/335908e3-d0bd-41c2-9d99-d10528a2d719)|

</details>

<details>
<summary>EliGen: 精準的影像分割槽控制</summary>

> 如何用分割槽圖層控制畫面內容的位置？

- 論文：[EliGen: Entity-Level Controlled Image Generation with Regional Attention](https://arxiv.org/abs/2501.01097)
- 程式碼樣例：[/examples/flux/model_inference/FLUX.1-dev-EliGen.py](/examples/flux/model_inference/FLUX.1-dev-EliGen.py)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen), [HuggingFace](https://huggingface.co/modelscope/EliGen)
- 線上體驗：[ModelScope EliGen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/EliGen)
- 資料集：[EliGen Train Set](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet)

|實體控制區域|生成影像|
|-|-|
|![eligen_example_2_mask_0](https://github.com/user-attachments/assets/1c6d9445-5022-4d91-ad2e-dc05321883d1)|![eligen_example_2_0](https://github.com/user-attachments/assets/86739945-cb07-4a49-b3b3-3bb65c90d14f)|

</details>

<details>
<summary>ExVideo: 影片生成模型的擴充套件訓練</summary>

> 如果影片生成模型只能生成 25 幀，如何才能讓它生成更長影片？

- 專案頁面：[Project Page](https://ecnu-cilab.github.io/ExVideoProjectPage/)
- 論文：[ExVideo: Extending Video Diffusion Models via Parameter-Efficient Post-Tuning](https://arxiv.org/abs/2406.14130)
- 程式碼樣例：請前往[舊版本](https://github.com/modelscope/DiffSynth-Studio/tree/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3/examples/ExVideo)檢視
- 模型：[ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1)

https://github.com/modelscope/DiffSynth-Studio/assets/35051019/d97f6aa9-8064-4b5b-9d49-ed6001bb9acc

</details>

<details>
<summary>Diffutoon: 高解析度動漫風格影片渲染</summary>

> 我管你這那的，我就是喜歡二次元！

- 專案頁面：[Project Page](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
- 論文：[Diffutoon: High-Resolution Editable Toon Shading via Diffusion Models](https://arxiv.org/abs/2401.16224)
- 程式碼樣例：請前往[舊版本](https://github.com/modelscope/DiffSynth-Studio/tree/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3/examples/Diffutoon)檢視

注意：Diffutoon 示例來自 2.0 之前的程式碼庫，目前不包含在 `main` 分支中。請將上面的舊版本程式碼與對應版本的 DiffSynth-Studio 一起使用。

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/b54c05c5-d747-4709-be5e-b39af82404dd

</details>

<details>
<summary>DiffSynth: 本專案的初代版本</summary>

> 在沒有影片生成模型的時代，如何利用影像生成模型處理影片？

- 專案頁面：[Project Page](https://ecnu-cilab.github.io/DiffSynth.github.io/)
- 論文：[DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis](https://arxiv.org/abs/2308.03463)
- 程式碼樣例：請前往[舊版本](https://github.com/modelscope/DiffSynth-Studio/tree/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3/examples/diffsynth)檢視

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/59fb2f7b-8de0-4481-b79f-0c3a7361a1ea

</details>

## 聯絡我們

|Discord：https://discord.gg/Mm9suEeUDc|
|-|
|<img width="160" height="160" alt="Image" src="https://github.com/user-attachments/assets/29bdc97b-e35d-4fea-88d6-32e35182e458" />|

## 全部支援的模型

| 基礎架構 | 模型 ID | 推理 | 低視訊記憶體推理 | 全量訓練 | 全量訓練後驗證 | LoRA 訓練 | LoRA 訓練後驗證 |
|-|-|-|-|-|-|-|-|
| Fibo 2.0 | [briaai/Fibo-2.0-dev](https://www.modelscope.cn/models/briaai/Fibo-2.0-dev) | [code](/examples/fibo/model_inference/Fibo-2.0-dev.py) | [code](/examples/fibo/model_inference_low_vram/Fibo-2.0-dev.py) | [code](/examples/fibo/model_training/full/Fibo-2.0-dev-T2I.sh) | [code](/examples/fibo/model_training/validate_full/Fibo-2.0-dev-T2I.py) | [code](/examples/fibo/model_training/lora/Fibo-2.0-dev-T2I.sh) | [code](/examples/fibo/model_training/validate_lora/Fibo-2.0-dev-T2I.py) |
| Fibo 2.0 | [briaai/Fibo-2.0-edit-turbo](https://www.modelscope.cn/models/briaai/Fibo-2.0-edit-turbo) | [code](/examples/fibo/model_inference/Fibo-2.0-edit-turbo.py) | [code](/examples/fibo/model_inference_low_vram/Fibo-2.0-edit-turbo.py) | - | - | - | - |
| MOSS-Image | [openmoss/MOSS-Image-Base](https://www.modelscope.cn/models/openmoss/MOSS-Image-Base) | [code](/examples/moss_image/model_inference/MOSS-Image-Base.py) | [code](/examples/moss_image/model_inference_low_vram/MOSS-Image-Base.py) | [code](/examples/moss_image/model_training/full/MOSS-Image-Base.sh) | [code](/examples/moss_image/model_training/validate_full/MOSS-Image-Base.py) | [code](/examples/moss_image/model_training/lora/MOSS-Image-Base.sh) | [code](/examples/moss_image/model_training/validate_lora/MOSS-Image-Base.py) |
| MOSS-Image | [openmoss/MOSS-Image-SFT](https://www.modelscope.cn/models/openmoss/MOSS-Image-SFT) | [code](/examples/moss_image/model_inference/MOSS-Image-SFT.py) | [code](/examples/moss_image/model_inference_low_vram/MOSS-Image-SFT.py) | [code](/examples/moss_image/model_training/full/MOSS-Image-SFT.sh) | [code](/examples/moss_image/model_training/validate_full/MOSS-Image-SFT.py) | [code](/examples/moss_image/model_training/lora/MOSS-Image-SFT.sh) | [code](/examples/moss_image/model_training/validate_lora/MOSS-Image-SFT.py) |
| MOSS-Image | [DiffSynth-Studio/MOSS-Image-EntroPack](https://www.modelscope.cn/models/DiffSynth-Studio/MOSS-Image-EntroPack) | [code](/examples/moss_image/model_inference/MOSS-Image-EntroPack.py) | [code](/examples/moss_image/model_inference_low_vram/MOSS-Image-EntroPack.py) | - | - | [code](/examples/moss_image/model_training/lora/MOSS-Image-entropack.sh) | [code](/examples/moss_image/model_training/validate_lora/MOSS-Image-entropack.py) |
| FLUX.3 | [black-forest-labs/flux-3-image-klein](https://www.modelscope.cn/models/black-forest-labs/flux-3-image-klein) | [code](/examples/flux3/model_inference/flux-3-image-klein.py) | [code](/examples/flux3/model_inference_low_vram/flux-3-image-klein.py) | [code](/examples/flux3/model_training/full/flux-3-image-klein.sh) | [code](/examples/flux3/model_training/validate_full/flux-3-image-klein.py) | [code](/examples/flux3/model_training/lora/flux-3-image-klein.sh) | [code](/examples/flux3/model_training/validate_lora/flux-3-image-klein.py) |
| FLUX.3 | [black-forest-labs/flux-3-image-klein-base](https://www.modelscope.cn/models/black-forest-labs/flux-3-image-klein-base) | [code](/examples/flux3/model_inference/flux-3-image-klein-base.py) | [code](/examples/flux3/model_inference_low_vram/flux-3-image-klein-base.py) | [code](/examples/flux3/model_training/full/flux-3-image-klein-base.sh) | [code](/examples/flux3/model_training/validate_full/flux-3-image-klein-base.py) | [code](/examples/flux3/model_training/lora/flux-3-image-klein-base.sh) | [code](/examples/flux3/model_training/validate_lora/flux-3-image-klein-base.py) |
| FLUX.3 | [DiffSynth-Studio/flux-3-image-klein-entropack](https://www.modelscope.cn/models/DiffSynth-Studio/flux-3-image-klein-entropack) | [code](/examples/flux3/model_inference/flux-3-image-klein-EntroPack.py) | [code](/examples/flux3/model_inference_low_vram/flux-3-image-klein-EntroPack.py) | - | - | [code](/examples/flux3/model_training/lora/flux-3-image-klein-entropack.sh) | [code](/examples/flux3/model_training/validate_lora/flux-3-image-klein-entropack.py) |
| Qwen-Image-2.1 | [Qwen/Qwen-Image-2.1](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1) | [code](/examples/qwen_image_21/model_inference/Qwen-Image-2.1.py) | [code](/examples/qwen_image_21/model_inference_low_vram/Qwen-Image-2.1.py) | [code](/examples/qwen_image_21/model_training/full/Qwen-Image-2.1.sh) | [code](/examples/qwen_image_21/model_training/validate_full/Qwen-Image-2.1.py) | [code](/examples/qwen_image_21/model_training/lora/Qwen-Image-2.1.sh) | [code](/examples/qwen_image_21/model_training/validate_lora/Qwen-Image-2.1.py) |
| Qwen-Image-2.1 | [Qwen/Qwen-Image-2.1-Turbo](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo) | [code](/examples/qwen_image_21/model_inference/Qwen-Image-2.1-Turbo.py) | [code](/examples/qwen_image_21/model_inference_low_vram/Qwen-Image-2.1-Turbo.py) | [code](/examples/qwen_image_21/model_training/full/Qwen-Image-2.1-Turbo.sh) | [code](/examples/qwen_image_21/model_training/validate_full/Qwen-Image-2.1-Turbo.py) | [code](/examples/qwen_image_21/model_training/lora/Qwen-Image-2.1-Turbo.sh) | [code](/examples/qwen_image_21/model_training/validate_lora/Qwen-Image-2.1-Turbo.py) |
| Qwen-Image-2.1 | [PAI/Qwen-Image-2.1-Fun-Controlnet-Union](https://www.modelscope.cn/models/PAI/Qwen-Image-2.1-Fun-Controlnet-Union) | [code](/examples/qwen_image_21/model_inference/Qwen-Image-2.1-Fun-Controlnet-Union.py) | [code](/examples/qwen_image_21/model_inference_low_vram/Qwen-Image-2.1-Fun-Controlnet-Union.py) | [code](/examples/qwen_image_21/model_training/full/Qwen-Image-2.1-Fun-Controlnet-Union.sh) | [code](/examples/qwen_image_21/model_training/validate_full/Qwen-Image-2.1-Fun-Controlnet-Union.py) | [code](/examples/qwen_image_21/model_training/lora/Qwen-Image-2.1-Fun-Controlnet-Union.sh) | [code](/examples/qwen_image_21/model_training/validate_lora/Qwen-Image-2.1-Fun-Controlnet-Union.py) |
| SenseNova-U1 | [SenseNova/SenseNova-U1.5-8B-MoT: T2I](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT) | [code](/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT.py) | [code](/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT.py) | [code](/examples/sensenova_u1/model_training/full/SenseNova-U1.5-8B-MoT.sh) | [code](/examples/sensenova_u1/model_training/validate_full/SenseNova-U1.5-8B-MoT.py) | [code](/examples/sensenova_u1/model_training/lora/SenseNova-U1.5-8B-MoT.sh) | [code](/examples/sensenova_u1/model_training/validate_lora/SenseNova-U1.5-8B-MoT.py) |
| SenseNova-U1 | [SenseNova/SenseNova-U1.5-8B-MoT: Edit](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT) | [code](/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT-Edit.py) | [code](/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT-Edit.py) | [code](/examples/sensenova_u1/model_training/full/SenseNova-U1.5-8B-MoT-Edit.sh) | [code](/examples/sensenova_u1/model_training/validate_full/SenseNova-U1.5-8B-MoT-Edit.py) | [code](/examples/sensenova_u1/model_training/lora/SenseNova-U1.5-8B-MoT-Edit.sh) | [code](/examples/sensenova_u1/model_training/validate_lora/SenseNova-U1.5-8B-MoT-Edit.py) |
| SenseNova-U1 | [SenseNova/SenseNova-U1.5-8B-MoT-LoRAs: 8-step](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-LoRAs) | [code](/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT-LoRA-8step.py) | [code](/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT-LoRA-8step.py) | - | - | - | - |
| SenseNova-U1 | [SenseNova/SenseNova-U1.5-8B-MoT-SFT: T2I](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-SFT) | [code](/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT-SFT.py) | [code](/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT-SFT.py) | [code](/examples/sensenova_u1/model_training/full/SenseNova-U1.5-8B-MoT-SFT.sh) | [code](/examples/sensenova_u1/model_training/validate_full/SenseNova-U1.5-8B-MoT-SFT.py) | [code](/examples/sensenova_u1/model_training/lora/SenseNova-U1.5-8B-MoT-SFT.sh) | [code](/examples/sensenova_u1/model_training/validate_lora/SenseNova-U1.5-8B-MoT-SFT.py) |
| SenseNova-U1 | [SenseNova/SenseNova-U1.5-8B-MoT-SFT: Edit](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-SFT) | [code](/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT-SFT-Edit.py) | [code](/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT-SFT-Edit.py) | [code](/examples/sensenova_u1/model_training/full/SenseNova-U1.5-8B-MoT-SFT-Edit.sh) | [code](/examples/sensenova_u1/model_training/validate_full/SenseNova-U1.5-8B-MoT-SFT-Edit.py) | [code](/examples/sensenova_u1/model_training/lora/SenseNova-U1.5-8B-MoT-SFT-Edit.sh) | [code](/examples/sensenova_u1/model_training/validate_lora/SenseNova-U1.5-8B-MoT-SFT-Edit.py) |
| MiniMax-Music3 | [MiniMax/MiniMax-Music3](https://www.modelscope.cn/models/MiniMax/MiniMax-Music3) | [code](/examples/minimax_music3/model_inference/MiniMax-Music3.py) | [code](/examples/minimax_music3/model_inference_low_vram/MiniMax-Music3.py) | — | — | — | — |
| YuE2 | [m-a-p/YuE2-3B](https://www.modelscope.cn/models/m-a-p/YuE2-3B) | [code](/examples/yue2/model_inference/YuE2.py) | [code](/examples/yue2/model_inference_low_vram/YuE2.py) | — | — | — | — |
| DiffSynth-Music | [DiffSynth-Studio/DiffSynth-Music](https://www.modelscope.cn/models/DiffSynth-Studio/DiffSynth-Music) | [code](/examples/diffsynth_music/model_inference/DiffSynth-Music.py) | [code](/examples/diffsynth_music/model_inference_low_vram/DiffSynth-Music.py) | [code](/examples/diffsynth_music/model_training/full/DiffSynth-Music.sh) | [code](/examples/diffsynth_music/model_training/validate_full/DiffSynth-Music.py) | - | - |
| MiniMax-H3 | [MiniMax/MiniMax-H3: FL2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FL2VA.py) |
| MiniMax-H3 | [MiniMax/MiniMax-H3: Ref2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Ref2VA.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Ref2VA.py) | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Ref2VA.py) |
| MiniMax-H3 | [MiniMax/MiniMax-H3: Retake](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Retake.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Retake.py) | - | - | - | - |
| MiniMax-H3 | [DiffSynth-Studio/MiniMax-H3-NF4: FL2VA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-NF4-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-FL2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-FL2VA.py) |
| MiniMax-H3 | [DiffSynth-Studio/MiniMax-H3-NF4: Ref2VA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-NF4-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-Ref2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-Ref2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: FL2VA pruned](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-Pruned-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Pruned-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Pruned-FL2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: Ref2VA pruned](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Pruned-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Pruned-Ref2VA.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-Pruned-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Pruned-Ref2VA.py) | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Pruned-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Pruned-Ref2VA.py) |
| MiniMax-H3 | [DiffSynth-Studio/MiniMax-H3-NF4: FL2VA pruned](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-NF4-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-Pruned-FL2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Pruned-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-Pruned-FL2VA.py) |
| MiniMax-H3 | [DiffSynth-Studio/MiniMax-H3-NF4: Ref2VA pruned](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-NF4-Pruned-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-Pruned-Ref2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Pruned-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-Pruned-Ref2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: FL2VA int8_convrot](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Int8-ConvRot-FL2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Int8-ConvRot-FL2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: Ref2VA int8_convrot](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Int8-ConvRot-Ref2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Int8-ConvRot-Ref2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: FL2VA pruned int8_convrot](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: Ref2VA pruned int8_convrot](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-Pruned-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Int8-ConvRot-Pruned-Ref2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-Pruned-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Int8-ConvRot-Pruned-Ref2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: FL2VA pruned fp8](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-FP8-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FP8-Pruned-FL2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-FP8-Pruned-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FP8-Pruned-FL2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: Ref2VA pruned fp8](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-FP8-Pruned-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FP8-Pruned-Ref2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-FP8-Pruned-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FP8-Pruned-Ref2VA.py) |
| MiniMax-H3 | [lightx2v/Minimax-h3-Turbo: FL2VA 4steps](https://www.modelscope.cn/models/lightx2v/Minimax-h3-Turbo) | [code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-FL2VA-Turbo.py) | [code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FL2VA-Turbo.py) | - | - | - | - |
| MiniMax-H3 | [DiffSynth-Studio/MiniMax-H3-Text-Embeddings](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-Text-Embeddings) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Text-Embeddings.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Text-Embeddings.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-Text-Embeddings.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Text-Embeddings.py) | - | - |
| MiniMax-H3 | [PAI/MiniMax-H3-Fun-Controlnet-Union](https://www.modelscope.cn/models/PAI/MiniMax-H3-Fun-Controlnet-Union) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Fun-Controlnet-Union.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Fun-Controlnet-Union.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-Fun-Controlnet-Union.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Fun-Controlnet-Union.py) | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Fun-Controlnet-Union.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Fun-Controlnet-Union.py) |
| MiniMax-H3 | [OpenVDN/vdn-minimax-h3: VDN-H3 50 步](https://www.modelscope.cn/models/OpenVDN/vdn-minimax-h3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-VDN-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-VDN-FL2VA.py) | - | - | - | - |
| MiniMax-H3 | [OpenVDN/vdn-minimax-h3: VDN-H3 8 步](https://www.modelscope.cn/models/OpenVDN/vdn-minimax-h3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-VDN-Turbo-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-VDN-Turbo-FL2VA.py) | - | - | - | - |
| MiniMax-H3 | [OpenVDN/vdn-minimax-h3: VDN-H3 FP8](https://www.modelscope.cn/models/OpenVDN/vdn-minimax-h3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-VDN-FP8-FL2VA.py) | - | - | - | - | - |
| MiniMax-H3 | [OpenVDN/vdn-minimax-h3: VDN-H3 8 步 FP8](https://www.modelscope.cn/models/OpenVDN/vdn-minimax-h3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-VDN-Turbo-FP8-FL2VA.py) | - | - | - | - | - |
| LingBot-Video | [Robbyant/lingbot-video-dense-1.3b: T2V](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_t2v.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b_t2v.py) | [code](/examples/lingbot_video/model_training/full/lingbot-video-dense-1.3b_t2v.sh) | [code](/examples/lingbot_video/model_training/validate_full/lingbot-video-dense-1.3b_t2v.py) | [code](/examples/lingbot_video/model_training/lora/lingbot-video-dense-1.3b_t2v.sh) | [code](/examples/lingbot_video/model_training/validate_lora/lingbot-video-dense-1.3b_t2v.py) |
| LingBot-Video | [Robbyant/lingbot-video-dense-1.3b: TI2V](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_ti2v.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b_ti2v.py) | [code](/examples/lingbot_video/model_training/full/lingbot-video-dense-1.3b_ti2v.sh) | [code](/examples/lingbot_video/model_training/validate_full/lingbot-video-dense-1.3b_ti2v.py) | [code](/examples/lingbot_video/model_training/lora/lingbot-video-dense-1.3b_ti2v.sh) | [code](/examples/lingbot_video/model_training/validate_lora/lingbot-video-dense-1.3b_ti2v.py) |
| LingBot-Video | [Robbyant/lingbot-video-dense-1.3b: T2I](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_t2i.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b_t2i.py) | - | - | - | - |
| LingBot-Video | [Robbyant/lingbot-video-moe-30b-a3b: T2V](https://modelscope.cn/models/Robbyant/lingbot-video-moe-30b-a3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-moe-30b-a3b_t2v.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-moe-30b-a3b_t2v.py) | [code](/examples/lingbot_video/model_training/full/lingbot-video-moe-30b-a3b_t2v.sh) | [code](/examples/lingbot_video/model_training/validate_full/lingbot-video-moe-30b-a3b_t2v.py) | [code](/examples/lingbot_video/model_training/lora/lingbot-video-moe-30b-a3b_t2v.sh) | [code](/examples/lingbot_video/model_training/validate_lora/lingbot-video-moe-30b-a3b_t2v.py) |
| LingBot-Video | [Robbyant/lingbot-video-moe-30b-a3b: TI2V](https://modelscope.cn/models/Robbyant/lingbot-video-moe-30b-a3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-moe-30b-a3b_ti2v.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-moe-30b-a3b_ti2v.py) | [code](/examples/lingbot_video/model_training/full/lingbot-video-moe-30b-a3b_ti2v.sh) | [code](/examples/lingbot_video/model_training/validate_full/lingbot-video-moe-30b-a3b_ti2v.py) | [code](/examples/lingbot_video/model_training/lora/lingbot-video-moe-30b-a3b_ti2v.sh) | [code](/examples/lingbot_video/model_training/validate_lora/lingbot-video-moe-30b-a3b_ti2v.py) |
| LingBot-Video | [Robbyant/lingbot-video-moe-30b-a3b: T2I](https://modelscope.cn/models/Robbyant/lingbot-video-moe-30b-a3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-moe-30b-a3b_t2i.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-moe-30b-a3b_t2i.py) | - | - | - | - |
| LingBot-Video | [Robbyant/lingbot-video-moe-30b-a3b: T2V + Refinement](https://modelscope.cn/models/Robbyant/lingbot-video-moe-30b-a3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-moe-30b-a3b_t2v_refiner.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-moe-30b-a3b_t2v_refiner.py) | - | - | - | - |
| LingBot-Video | [Robbyant/lingbot-video-moe-30b-a3b: TI2V + Refinement](https://modelscope.cn/models/Robbyant/lingbot-video-moe-30b-a3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-moe-30b-a3b_ti2v_refiner.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-moe-30b-a3b_ti2v_refiner.py) | - | - | - | - |
| ACE-Step | [ACE-Step/Ace-Step1.5](https://www.modelscope.cn/models/ACE-Step/Ace-Step1.5) | [code](/examples/ace_step/model_inference/Ace-Step1.5.py) | [code](/examples/ace_step/model_inference_low_vram/Ace-Step1.5.py) | [code](/examples/ace_step/model_training/full/Ace-Step1.5.sh) | [code](/examples/ace_step/model_training/validate_full/Ace-Step1.5.py) | [code](/examples/ace_step/model_training/lora/Ace-Step1.5.sh) | [code](/examples/ace_step/model_training/validate_lora/Ace-Step1.5.py) |
| ACE-Step | [ACE-Step/acestep-v15-turbo-shift1](https://www.modelscope.cn/models/ACE-Step/acestep-v15-turbo-shift1) | [code](/examples/ace_step/model_inference/acestep-v15-turbo-shift1.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-turbo-shift1.py) | [code](/examples/ace_step/model_training/full/acestep-v15-turbo-shift1.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-turbo-shift1.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-turbo-shift1.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-turbo-shift1.py) |
| ACE-Step | [ACE-Step/acestep-v15-turbo-shift3](https://www.modelscope.cn/models/ACE-Step/acestep-v15-turbo-shift3) | [code](/examples/ace_step/model_inference/acestep-v15-turbo-shift3.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-turbo-shift3.py) | [code](/examples/ace_step/model_training/full/acestep-v15-turbo-shift3.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-turbo-shift3.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-turbo-shift3.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-turbo-shift3.py) |
| ACE-Step | [ACE-Step/acestep-v15-turbo-continuous](https://www.modelscope.cn/models/ACE-Step/acestep-v15-turbo-continuous) | [code](/examples/ace_step/model_inference/acestep-v15-turbo-continuous.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-turbo-continuous.py) | [code](/examples/ace_step/model_training/full/acestep-v15-turbo-continuous.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-turbo-continuous.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-turbo-continuous.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-turbo-continuous.py) |
| ACE-Step | [ACE-Step/acestep-v15-base](https://www.modelscope.cn/models/ACE-Step/acestep-v15-base) | [code](/examples/ace_step/model_inference/acestep-v15-base.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-base.py) | [code](/examples/ace_step/model_training/full/acestep-v15-base.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-base.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-base.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-base.py) |
| ACE-Step | [ACE-Step/acestep-v15-base: CoverTask](https://www.modelscope.cn/models/ACE-Step/acestep-v15-base) | [code](/examples/ace_step/model_inference/acestep-v15-base-CoverTask.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-base-CoverTask.py) | — | — | — | — |
| ACE-Step | [ACE-Step/acestep-v15-base: RepaintTask](https://www.modelscope.cn/models/ACE-Step/acestep-v15-base) | [code](/examples/ace_step/model_inference/acestep-v15-base-RepaintTask.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-base-RepaintTask.py) | — | — | — | — |
| ACE-Step | [ACE-Step/acestep-v15-sft](https://www.modelscope.cn/models/ACE-Step/acestep-v15-sft) | [code](/examples/ace_step/model_inference/acestep-v15-sft.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-sft.py) | [code](/examples/ace_step/model_training/full/acestep-v15-sft.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-sft.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-sft.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-sft.py) |
| ACE-Step | [ACE-Step/acestep-v15-xl-base](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-base) | [code](/examples/ace_step/model_inference/acestep-v15-xl-base.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-xl-base.py) | [code](/examples/ace_step/model_training/full/acestep-v15-xl-base.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-xl-base.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-xl-base.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-xl-base.py) |
| ACE-Step | [ACE-Step/acestep-v15-xl-sft](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-sft) | [code](/examples/ace_step/model_inference/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_training/full/acestep-v15-xl-sft.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-xl-sft.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-xl-sft.py) |
| ACE-Step | [ACE-Step/acestep-v15-xl-turbo](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-turbo) | [code](/examples/ace_step/model_inference/acestep-v15-xl-turbo.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-xl-turbo.py) | [code](/examples/ace_step/model_training/full/acestep-v15-xl-turbo.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-xl-turbo.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-xl-turbo.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-xl-turbo.py) |
| ACE-Step | [DiffSynth-Studio/acestep15xlsft-lora-music](https://www.modelscope.cn/models/DiffSynth-Studio/acestep15xlsft-lora-music) | [code](/examples/ace_step/model_inference/acestep15xlsft-vocals2music.py) | [code](/examples/ace_step/model_inference_low_vram/acestep15xlsft-vocals2music.py) | [code](/examples/ace_step/model_training/full/acestep15xlsft-vocals2music.sh) | [code](/examples/ace_step/model_training/validate_full/acestep15xlsft-vocals2music.py) | - | - |
| Boogu-Image | [Boogu/Boogu-Image-0.1-Base](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Base) | [code](/examples/boogu_image/model_inference/Boogu-Image-0.1-Base.py) | [code](/examples/boogu_image/model_inference_low_vram/Boogu-Image-0.1-Base.py) | [code](/examples/boogu_image/model_training/full/Boogu-Image-0.1-Base.sh) | [code](/examples/boogu_image/model_training/validate_full/Boogu-Image-0.1-Base.py) | [code](/examples/boogu_image/model_training/lora/Boogu-Image-0.1-Base.sh) | [code](/examples/boogu_image/model_training/validate_lora/Boogu-Image-0.1-Base.py) |
| Boogu-Image | [Boogu/Boogu-Image-0.1-Turbo](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Turbo) | [code](/examples/boogu_image/model_inference/Boogu-Image-0.1-Turbo.py) | [code](/examples/boogu_image/model_inference_low_vram/Boogu-Image-0.1-Turbo.py) | [code](/examples/boogu_image/model_training/full/Boogu-Image-0.1-Turbo.sh) | [code](/examples/boogu_image/model_training/validate_full/Boogu-Image-0.1-Turbo.py) | [code](/examples/boogu_image/model_training/lora/Boogu-Image-0.1-Turbo.sh) | [code](/examples/boogu_image/model_training/validate_lora/Boogu-Image-0.1-Turbo.py) |
| Boogu-Image | [Boogu/Boogu-Image-0.1-Edit](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Edit) | [code](/examples/boogu_image/model_inference/Boogu-Image-0.1-Edit.py) | [code](/examples/boogu_image/model_inference_low_vram/Boogu-Image-0.1-Edit.py) | [code](/examples/boogu_image/model_training/full/Boogu-Image-0.1-Edit.sh) | [code](/examples/boogu_image/model_training/validate_full/Boogu-Image-0.1-Edit.py) | [code](/examples/boogu_image/model_training/lora/Boogu-Image-0.1-Edit.sh) | [code](/examples/boogu_image/model_training/validate_lora/Boogu-Image-0.1-Edit.py) |
| Krea-2 | [krea/Krea-2-Raw](https://www.modelscope.cn/models/krea/Krea-2-Raw) | [code](/examples/krea2/model_inference/Krea-2-Raw.py) | [code](/examples/krea2/model_inference_low_vram/Krea-2-Raw.py) | [code](/examples/krea2/model_training/full/Krea-2-Raw.sh) | [code](/examples/krea2/model_training/validate_full/Krea-2-Raw.py) | [code](/examples/krea2/model_training/lora/Krea-2-Raw.sh) | [code](/examples/krea2/model_training/validate_lora/Krea-2-Raw.py) |
| Krea-2 | [krea/Krea-2-Turbo](https://www.modelscope.cn/models/krea/Krea-2-Turbo) | [code](/examples/krea2/model_inference/Krea-2-Turbo.py) | [code](/examples/krea2/model_inference_low_vram/Krea-2-Turbo.py) | [code](/examples/krea2/model_training/full/Krea-2-Turbo.sh) | [code](/examples/krea2/model_training/validate_full/Krea-2-Turbo.py) | [code](/examples/krea2/model_training/lora/Krea-2-Turbo.sh) | [code](/examples/krea2/model_training/validate_lora/Krea-2-Turbo.py) |
| Ideogram 4 | [ideogram-ai/ideogram-4-fp8](https://www.modelscope.cn/models/ideogram-ai/ideogram-4-fp8) | [code](/examples/ideogram4/model_inference/ideogram-4-fp8.py) | - | - | - | - | - |
| Ideogram 4 | [DiffSynth-Studio/ideogram-4-bf16-repackage](https://www.modelscope.cn/models/DiffSynth-Studio/ideogram-4-bf16-repackage) | [code](/examples/ideogram4/model_inference/ideogram-4-bf16-repackage.py) | [code](/examples/ideogram4/model_inference_low_vram/ideogram-4-bf16-repackage.py) | [code](/examples/ideogram4/model_training/full/Ideogram-4-bf16-repackage.sh) | - | [code](/examples/ideogram4/model_training/lora/Ideogram-4-bf16-repackage.sh) | [code](/examples/ideogram4/model_training/validate_lora/Ideogram-4-bf16-repackage.py) |
| HiDream-O1-Image | [HiDream-ai/HiDream-O1-Image](https://modelscope.cn/models/HiDream-ai/HiDream-O1-Image) | [code](/examples/hidream_o1_image/model_inference/HiDream-O1-Image.py) | [code](/examples/hidream_o1_image/model_inference_low_vram/HiDream-O1-Image.py) | [code](/examples/hidream_o1_image/model_training/full/HiDream-O1-Image.sh) | [code](/examples/hidream_o1_image/model_training/validate_full/HiDream-O1-Image.py) | [code](/examples/hidream_o1_image/model_training/lora/HiDream-O1-Image.sh) | [code](/examples/hidream_o1_image/model_training/validate_lora/HiDream-O1-Image.py) |
| HiDream-O1-Image | [HiDream-ai/HiDream-O1-Image-Dev](https://modelscope.cn/models/HiDream-ai/HiDream-O1-Image-Dev) | [code](/examples/hidream_o1_image/model_inference/HiDream-O1-Image-Dev.py) | [code](/examples/hidream_o1_image/model_inference_low_vram/HiDream-O1-Image-Dev.py) | [code](/examples/hidream_o1_image/model_training/full/HiDream-O1-Image-Dev.sh) | [code](/examples/hidream_o1_image/model_training/validate_full/HiDream-O1-Image-Dev.py) | [code](/examples/hidream_o1_image/model_training/lora/HiDream-O1-Image-Dev.sh) | [code](/examples/hidream_o1_image/model_training/validate_lora/HiDream-O1-Image-Dev.py) |
| HiDream-O1-Image | [DiffSynth-Studio/HidreamO1-i2L-v2](https://www.modelscope.cn/models/DiffSynth-Studio/HidreamO1-i2L-v2) | [code](/examples/hidream_o1_image/model_inference/HidreamO1-i2L-v2.py) | [code](/examples/hidream_o1_image/model_inference_low_vram/HidreamO1-i2L-v2.py) | [code](/examples/hidream_o1_image/model_training/full/HidreamO1-i2L-v2.sh) | [code](/examples/hidream_o1_image/model_training/validate_full/HidreamO1-i2L-v2.py) | - | - |
| JoyAI-Image | [jd-opensource/JoyAI-Image-Edit](https://modelscope.cn/models/jd-opensource/JoyAI-Image-Edit) | [code](/examples/joyai_image/model_inference/JoyAI-Image-Edit.py) | [code](/examples/joyai_image/model_inference_low_vram/JoyAI-Image-Edit.py) | [code](/examples/joyai_image/model_training/full/JoyAI-Image-Edit.sh) | [code](/examples/joyai_image/model_training/validate_full/JoyAI-Image-Edit.py) | [code](/examples/joyai_image/model_training/lora/JoyAI-Image-Edit.sh) | [code](/examples/joyai_image/model_training/validate_lora/JoyAI-Image-Edit.py) |
| ERNIE-Image | [PaddlePaddle/ERNIE-Image](https://www.modelscope.cn/models/PaddlePaddle/ERNIE-Image) | [code](/examples/ernie_image/model_inference/ERNIE-Image.py) | [code](/examples/ernie_image/model_inference_low_vram/ERNIE-Image.py) | [code](/examples/ernie_image/model_training/full/ERNIE-Image.sh) | [code](/examples/ernie_image/model_training/validate_full/ERNIE-Image.py) | [code](/examples/ernie_image/model_training/lora/ERNIE-Image.sh) | [code](/examples/ernie_image/model_training/validate_lora/ERNIE-Image.py) |
| ERNIE-Image | [PaddlePaddle/ERNIE-Image-Turbo](https://www.modelscope.cn/models/PaddlePaddle/ERNIE-Image-Turbo) | [code](/examples/ernie_image/model_inference/ERNIE-Image-Turbo.py) | [code](/examples/ernie_image/model_inference_low_vram/ERNIE-Image-Turbo.py) | — | — | — | — |
| LTX-2.5 | [Lightricks/LTX-2.5: DistilledPipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.5) | [code](/examples/ltx2/model_inference/LTX-2.5-T2AV-DistilledPipeline.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-DistilledPipeline.py) | [code](/examples/ltx2/model_training/full/LTX-2.5-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_full/LTX-2.5-T2AV.py) | [code](/examples/ltx2/model_training/lora/LTX-2.5-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2.5-T2AV.py) |
| LTX-2.5 | [Lightricks/LTX-2.5: DistilledPipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.5) | [code](/examples/ltx2/model_inference/LTX-2.5-I2AV-DistilledPipeline.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.5-I2AV-DistilledPipeline.py) | - | - | - | - |
| LTX-2.5 | [Lightricks/LTX-2.5: TwoStagePipeline-A2V](https://www.modelscope.cn/models/Lightricks/LTX-2.5) | [code](/examples/ltx2/model_inference/LTX-2.5-A2V-TwoStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.5-A2V-TwoStage.py) | - | - | - | - |
| LTX-2.5 | [Lightricks/LTX-2.5: TwoStagePipeline-Retake](https://www.modelscope.cn/models/Lightricks/LTX-2.5) | [code](/examples/ltx2/model_inference/LTX-2.5-T2AV-TwoStage-Retake.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-TwoStage-Retake.py) | - | - | - | - |
| LTX-2.5 | [Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler](https://www.modelscope.cn/models/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler) | [code](/examples/ltx2/model_inference/LTX-2.5-IC-LoRA-Pixel-Spatial-Upscaler.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.5-IC-LoRA-Pixel-Spatial-Upscaler.py) | - | - | - | - |
| LTX-2.5 | [Lightricks/LTX-2.5: T2A](https://www.modelscope.cn/models/Lightricks/LTX-2.5) | [code](/examples/ltx2/model_inference/LTX-2.5-T2A.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2A.py) | - | - | - | - |
| LTX-2.5 | [Lightricks/LTX-2.5: INT8-ConvRot](https://www.modelscope.cn/models/Lightricks/LTX-2.5) | [code](/examples/ltx2/model_inference/LTX-2.5-T2AV-INT8-ConvRot.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-INT8-ConvRot.py) | - | - | [code](/examples/ltx2/model_training/lora/LTX-2.5-Int8-ConvRot-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2.5-Int8-ConvRot.py) |
| LTX-2 | [jd-opensource/JoyAI-Echo](https://modelscope.cn/models/jd-opensource/JoyAI-Echo) | [code](/examples/ltx2/model_inference/JoyAI-Echo-T2AV.py) | [code](/examples/ltx2/model_inference_low_vram/JoyAI-Echo-T2AV.py) | [code](/examples/ltx2/model_training/full/JoyAI-Echo-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_full/JoyAI-Echo-T2AV.py) | [code](/examples/ltx2/model_training/lora/JoyAI-Echo-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/JoyAI-Echo-T2AV.py) |
| LTX-2 | [Lightricks/LTX-2.3: OneStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-I2AV-OneStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-I2AV-OneStage.py) | [code](/examples/ltx2/model_training/full/LTX-2.3-I2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_full/LTX-2.3-I2AV.py) | [code](/examples/ltx2/model_training/lora/LTX-2.3-I2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2.3-I2AV.py) |
| LTX-2 | [Lightricks/LTX-2.3: TwoStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-I2AV-TwoStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-I2AV-TwoStage.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2.3: DistilledPipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-I2AV-DistilledPipeline.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-I2AV-DistilledPipeline.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2.3: OneStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-T2AV-OneStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-T2AV-OneStage.py) | [code](/examples/ltx2/model_training/full/LTX-2.3-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_full/LTX-2.3-T2AV.py) | [code](/examples/ltx2/model_training/lora/LTX-2.3-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2.3-T2AV.py) |
| LTX-2 | [Lightricks/LTX-2.3: TwoStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-T2AV-TwoStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-T2AV-TwoStage.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2.3: DistilledPipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-T2AV-DistilledPipeline.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-T2AV-DistilledPipeline.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2.3: A2V](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-A2V-TwoStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-A2V-TwoStage.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2.3: Retake](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-T2AV-TwoStage-Retake.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-T2AV-TwoStage-Retake.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://www.modelscope.cn/models/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) | [code](/examples/ltx2/model_inference/LTX-2.3-T2AV-IC-LoRA-Union-Control.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-T2AV-IC-LoRA-Union-Control.py) | - | - | [code](/examples/ltx2/model_training/lora/LTX-2.3-T2AV-IC-LoRA-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2.3-T2AV-IC-LoRA.py) |
| LTX-2 | [Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control](https://www.modelscope.cn/models/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control) | [code](/examples/ltx2/model_inference/LTX-2.3-T2AV-IC-LoRA-Motion-Track-Control.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-T2AV-IC-LoRA-Motion-Track-Control.py) | - | - | [code](/examples/ltx2/model_training/lora/LTX-2.3-T2AV-IC-LoRA-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2.3-T2AV-IC-LoRA.py) |
| LTX-2 | [Lightricks/LTX-2: OneStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-OneStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-OneStage.py) | [code](/examples/ltx2/model_training/full/LTX-2-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_full/LTX-2-T2AV.py) | [code](/examples/ltx2/model_training/lora/LTX-2-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2-T2AV.py) |
| LTX-2 | [Lightricks/LTX-2-19b-IC-LoRA-Union-Control](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-IC-LoRA-Union-Control) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-IC-LoRA-Union-Control.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-IC-LoRA-Union-Control.py) | - | - | [code](/examples/ltx2/model_training/lora/LTX-2-T2AV-IC-LoRA-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2-T2AV-IC-LoRA.py) |
| LTX-2 | [Lightricks/LTX-2-19b-IC-LoRA-Detailer](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-IC-LoRA-Detailer) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-IC-LoRA-Detailer.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-IC-LoRA-Detailer.py) | - | - | [code](/examples/ltx2/model_training/lora/LTX-2-T2AV-IC-LoRA-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2-T2AV-IC-LoRA.py) |
| LTX-2 | [Lightricks/LTX-2: TwoStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-TwoStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-TwoStage.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2: DistilledPipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-DistilledPipeline.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-DistilledPipeline.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2: OneStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2) | [code](/examples/ltx2/model_inference/LTX-2-I2AV-OneStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-OneStage.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2: TwoStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2) | [code](/examples/ltx2/model_inference/LTX-2-I2AV-TwoStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-TwoStage.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2: DistilledPipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2) | [code](/examples/ltx2/model_inference/LTX-2-I2AV-DistilledPipeline.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-DistilledPipeline.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-In.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-In.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-Out.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-Out.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-Left.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-Left.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-Right.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-Right.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Jib-Up.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Jib-Up.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Jib-Down.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Jib-Down.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Static](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Static.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Static.py) | - | - | - | - |
| FLUX.2 | [black-forest-labs/FLUX.2-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-dev) | [code](/examples/flux2/model_inference/FLUX.2-dev.py) | [code](/examples/flux2/model_inference_low_vram/FLUX.2-dev.py) | - | - | [code](/examples/flux2/model_training/lora/FLUX.2-dev.sh) | [code](/examples/flux2/model_training/validate_lora/FLUX.2-dev.py) |
| FLUX.2 | [black-forest-labs/FLUX.2-klein-4B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B) | [code](/examples/flux2/model_inference/FLUX.2-klein-4B.py) | [code](/examples/flux2/model_inference_low_vram/FLUX.2-klein-4B.py) | [code](/examples/flux2/model_training/full/FLUX.2-klein-4B.sh) | [code](/examples/flux2/model_training/validate_full/FLUX.2-klein-4B.py) | [code](/examples/flux2/model_training/lora/FLUX.2-klein-4B.sh) | [code](/examples/flux2/model_training/validate_lora/FLUX.2-klein-4B.py) |
| FLUX.2 | [black-forest-labs/FLUX.2-klein-9B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-9B) | [code](/examples/flux2/model_inference/FLUX.2-klein-9B.py) | [code](/examples/flux2/model_inference_low_vram/FLUX.2-klein-9B.py) | [code](/examples/flux2/model_training/full/FLUX.2-klein-9B.sh) | [code](/examples/flux2/model_training/validate_full/FLUX.2-klein-9B.py) | [code](/examples/flux2/model_training/lora/FLUX.2-klein-9B.sh) | [code](/examples/flux2/model_training/validate_lora/FLUX.2-klein-9B.py) |
| FLUX.2 | [black-forest-labs/FLUX.2-klein-base-4B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-4B) | [code](/examples/flux2/model_inference/FLUX.2-klein-base-4B.py) | [code](/examples/flux2/model_inference_low_vram/FLUX.2-klein-base-4B.py) | [code](/examples/flux2/model_training/full/FLUX.2-klein-base-4B.sh) | [code](/examples/flux2/model_training/validate_full/FLUX.2-klein-base-4B.py) | [code](/examples/flux2/model_training/lora/FLUX.2-klein-base-4B.sh) | [code](/examples/flux2/model_training/validate_lora/FLUX.2-klein-base-4B.py) |
| FLUX.2 | [black-forest-labs/FLUX.2-klein-base-9B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-9B) | [code](/examples/flux2/model_inference/FLUX.2-klein-base-9B.py) | [code](/examples/flux2/model_inference_low_vram/FLUX.2-klein-base-9B.py) | [code](/examples/flux2/model_training/full/FLUX.2-klein-base-9B.sh) | [code](/examples/flux2/model_training/validate_full/FLUX.2-klein-base-9B.py) | [code](/examples/flux2/model_training/lora/FLUX.2-klein-base-9B.sh) | [code](/examples/flux2/model_training/validate_lora/FLUX.2-klein-base-9B.py) |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Aesthetic](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Aesthetic) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Aesthetic.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Aesthetic.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Aesthetic.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Aesthetic.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Brightness](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Brightness) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Brightness.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Brightness.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Brightness.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Brightness.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Age](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Age) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Age.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Age.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Age.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Age.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-ControlNet](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-ControlNet) | [code](/examples/flux2/model_inference/Template-KleinBase4B-ControlNet.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-ControlNet.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-ControlNet.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-ControlNet.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Edit](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Edit) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Edit.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Edit.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Edit.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Edit.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Inpaint](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Inpaint) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Inpaint.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Inpaint.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Inpaint.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Inpaint.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-PandaMeme](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-PandaMeme) | [code](/examples/flux2/model_inference/Template-KleinBase4B-PandaMeme.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-PandaMeme.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-PandaMeme.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-PandaMeme.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Sharpness](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Sharpness) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Sharpness.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Sharpness.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Sharpness.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Sharpness.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-SoftRGB](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-SoftRGB) | [code](/examples/flux2/model_inference/Template-KleinBase4B-SoftRGB.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-SoftRGB.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-SoftRGB.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-SoftRGB.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Upscaler](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Upscaler) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Upscaler.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Upscaler.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Upscaler.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Upscaler.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-ContentRef](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-ContentRef) | [code](/examples/flux2/model_inference/Template-KleinBase4B-ContentRef.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-ContentRef.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-ContentRef.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-ContentRef.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/KleinBase4B-i2L-v2](https://www.modelscope.cn/models/DiffSynth-Studio/KleinBase4B-i2L-v2) | [code](/examples/flux2/model_inference/KleinBase4B-i2L-v2.py) | [code](/examples/flux2/model_inference_low_vram/KleinBase4B-i2L-v2.py) | [code](/examples/flux2/model_training/full/KleinBase4B-i2L-v2.sh) | [code](/examples/flux2/model_training/validate_full/KleinBase4B-i2L-v2.py) | - | - |
| Z-Image | [Tongyi-MAI/Z-Image](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image) | [code](/examples/z_image/model_inference/Z-Image.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image.py) | [code](/examples/z_image/model_training/full/Z-Image.sh) | [code](/examples/z_image/model_training/validate_full/Z-Image.py) | [code](/examples/z_image/model_training/lora/Z-Image.sh) | [code](/examples/z_image/model_training/validate_lora/Z-Image.py) |
| Z-Image | [DiffSynth-Studio/Z-Image-i2L](https://www.modelscope.cn/models/DiffSynth-Studio/Z-Image-i2L) | [code](/examples/z_image/model_inference/Z-Image-i2L.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image-i2L.py) | - | - | - | - |
| Z-Image | [Tongyi-MAI/Z-Image-Turbo](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) | [code](/examples/z_image/model_inference/Z-Image-Turbo.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image-Turbo.py) | [code](/examples/z_image/model_training/full/Z-Image-Turbo.sh) | [code](/examples/z_image/model_training/validate_full/Z-Image-Turbo.py) | [code](/examples/z_image/model_training/lora/Z-Image-Turbo.sh) | [code](/examples/z_image/model_training/validate_lora/Z-Image-Turbo.py) |
| Z-Image | [PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1](https://www.modelscope.cn/models/PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | [code](/examples/z_image/model_inference/Z-Image-Turbo-Fun-Controlnet-Union-2.1.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image-Turbo-Fun-Controlnet-Union-2.1.py) | [code](/examples/z_image/model_training/full/Z-Image-Turbo-Fun-Controlnet-Union-2.1.sh) | [code](/examples/z_image/model_training/validate_full/Z-Image-Turbo-Fun-Controlnet-Union-2.1.py) | [code](/examples/z_image/model_training/lora/Z-Image-Turbo-Fun-Controlnet-Union-2.1.sh) | [code](/examples/z_image/model_training/validate_lora/Z-Image-Turbo-Fun-Controlnet-Union-2.1.py) |
| Z-Image | [PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps](https://www.modelscope.cn/models/PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | [code](/examples/z_image/model_inference/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.py) | [code](/examples/z_image/model_training/full/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.sh) | [code](/examples/z_image/model_training/validate_full/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.py) | [code](/examples/z_image/model_training/lora/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.sh) | [code](/examples/z_image/model_training/validate_lora/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.py) |
| Z-Image | [PAI/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps](https://www.modelscope.cn/models/PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | [code](/examples/z_image/model_inference/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.py) | [code](/examples/z_image/model_training/full/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.sh) | [code](/examples/z_image/model_training/validate_full/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.py) | [code](/examples/z_image/model_training/lora/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.sh) | [code](/examples/z_image/model_training/validate_lora/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.py) |
| Z-Image | [DiffSynth-Studio/ZImage-i2L-v2](https://www.modelscope.cn/models/DiffSynth-Studio/ZImage-i2L-v2) | [code](/examples/z_image/model_inference/ZImage-i2L-v2.py) | [code](/examples/z_image/model_inference_low_vram/ZImage-i2L-v2.py) | [code](/examples/z_image/model_training/full/ZImage-i2L-v2.sh) | [code](/examples/z_image/model_training/validate_full/ZImage-i2L-v2.py) | - | - |
| Anima | [circlestone-labs/Anima](https://www.modelscope.cn/models/circlestone-labs/Anima) | [code](/examples/anima/model_inference/anima-preview.py) | [code](/examples/anima/model_inference_low_vram/anima-preview.py) | [code](/examples/anima/model_training/full/anima-preview.sh) | [code](/examples/anima/model_training/validate_full/anima-preview.py) | [code](/examples/anima/model_training/lora/anima-preview.sh) | [code](/examples/anima/model_training/validate_lora/anima-preview.py) |
| Qwen-Image | [Qwen/Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image) | [code](/examples/qwen_image/model_inference/Qwen-Image.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image.py) |
| Qwen-Image | [Qwen/Qwen-Image-2512](https://www.modelscope.cn/models/Qwen/Qwen-Image-2512) | [code](/examples/qwen_image/model_inference/Qwen-Image-2512.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-2512.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-2512.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-2512.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-2512.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-2512.py) |
| Qwen-Image | [Qwen/Qwen-Image-Edit](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit) | [code](/examples/qwen_image/model_inference/Qwen-Image-Edit.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Edit.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Edit.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Edit.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit.py) |
| Qwen-Image | [Qwen/Qwen-Image-Edit-2509](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2509) | [code](/examples/qwen_image/model_inference/Qwen-Image-Edit-2509.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-2509.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Edit-2509.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Edit-2509.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2509.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2509.py) |
| Qwen-Image | [Qwen/Qwen-Image-Edit-2511](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2511) | [code](/examples/qwen_image/model_inference/Qwen-Image-Edit-2511.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-2511.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Edit-2511.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Edit-2511.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2511.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2511.py) |
| Qwen-Image | [FireRedTeam/FireRed-Image-Edit-1.0](https://www.modelscope.cn/models/FireRedTeam/FireRed-Image-Edit-1.0) | [code](/examples/qwen_image/model_inference/FireRed-Image-Edit-1.0.py) | [code](/examples/qwen_image/model_inference_low_vram/FireRed-Image-Edit-1.0.py) | [code](/examples/qwen_image/model_training/full/FireRed-Image-Edit-1.0.sh) | [code](/examples/qwen_image/model_training/validate_full/FireRed-Image-Edit-1.0.py) | [code](/examples/qwen_image/model_training/lora/FireRed-Image-Edit-1.0.sh) | [code](/examples/qwen_image/model_training/validate_lora/FireRed-Image-Edit-1.0.py) |
| Qwen-Image | [FireRedTeam/FireRed-Image-Edit-1.1](https://www.modelscope.cn/models/FireRedTeam/FireRed-Image-Edit-1.1) | [code](/examples/qwen_image/model_inference/FireRed-Image-Edit-1.1.py) | [code](/examples/qwen_image/model_inference_low_vram/FireRed-Image-Edit-1.1.py) | [code](/examples/qwen_image/model_training/full/FireRed-Image-Edit-1.1.sh) | [code](/examples/qwen_image/model_training/validate_full/FireRed-Image-Edit-1.1.py) | [code](/examples/qwen_image/model_training/lora/FireRed-Image-Edit-1.1.sh) | [code](/examples/qwen_image/model_training/validate_lora/FireRed-Image-Edit-1.1.py) |
| Qwen-Image | [lightx2v/Qwen-Image-Edit-2511-Lightning](https://modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning) | [code](/examples/qwen_image/model_inference/Qwen-Image-Edit-2511-Lightning.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-2511-Lightning.py) | - | - | - | - |
| Qwen-Image | [Qwen/Qwen-Image-Layered](https://www.modelscope.cn/models/Qwen/Qwen-Image-Layered) | [code](/examples/qwen_image/model_inference/Qwen-Image-Layered.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Layered.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Layered.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Layered.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Layered.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Layered.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Layered-Control](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Layered-Control) | [code](/examples/qwen_image/model_inference/Qwen-Image-Layered-Control.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Layered-Control.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Layered-Control.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Layered-Control.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Layered-Control.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Layered-Control.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Layered-Control-V2](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Layered-Control-V2) | [code](/examples/qwen_image/model_inference/Qwen-Image-Layered-Control-V2.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Layered-Control-V2.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Layered-Control-V2.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Layered-Control-V2.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-EliGen](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen) | [code](/examples/qwen_image/model_inference/Qwen-Image-EliGen.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-EliGen.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-EliGen.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-EliGen.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-EliGen-V2](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-V2) | [code](/examples/qwen_image/model_inference/Qwen-Image-EliGen-V2.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-EliGen-V2.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-EliGen.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-EliGen.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-EliGen-Poster](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-Poster) | [code](/examples/qwen_image/model_inference/Qwen-Image-EliGen-Poster.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-EliGen-Poster.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-EliGen-Poster.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-EliGen-Poster.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full) | [code](/examples/qwen_image/model_inference/Qwen-Image-Distill-Full.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Distill-Full.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Distill-Full.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Distill-Full.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Distill-Full.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Distill-Full.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA) | [code](/examples/qwen_image/model_inference/Qwen-Image-Distill-LoRA.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Distill-LoRA.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Distill-LoRA.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Distill-LoRA.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny) | [code](/examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Canny.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Canny.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Blockwise-ControlNet-Canny.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Canny.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Blockwise-ControlNet-Canny.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Canny.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth) | [code](/examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Depth.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Depth.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Blockwise-ControlNet-Depth.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Depth.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Blockwise-ControlNet-Depth.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Depth.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint) | [code](/examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Inpaint.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Inpaint.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Blockwise-ControlNet-Inpaint.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Inpaint.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Blockwise-ControlNet-Inpaint.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Inpaint.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-In-Context-Control-Union](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-In-Context-Control-Union) | [code](/examples/qwen_image/model_inference/Qwen-Image-In-Context-Control-Union.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-In-Context-Control-Union.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-In-Context-Control-Union.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-In-Context-Control-Union.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix) | [code](/examples/qwen_image/model_inference/Qwen-Image-Edit-Lowres-Fix.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-Lowres-Fix.py) | - | - | - | - |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-i2L](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-i2L) | [code](/examples/qwen_image/model_inference/Qwen-Image-i2L.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-i2L.py) | - | - | - | - |
| Qwen-Video-Edit | [yunpeng1998/Qwen-Video-Edit](https://www.modelscope.cn/models/yunpeng1998/Qwen-Video-Edit) | [code](/examples/qwen_video_edit/model_inference/Qwen-Video-Edit.py) | [code](/examples/qwen_video_edit/model_inference_low_vram/Qwen-Video-Edit.py) | [code](/examples/qwen_video_edit/model_training/full/Qwen-Video-Edit.sh) | [code](/examples/qwen_video_edit/model_training/validate_full/Qwen-Video-Edit.py) | [code](/examples/qwen_video_edit/model_training/lora/Qwen-Video-Edit.sh) | [code](/examples/qwen_video_edit/model_training/validate_lora/Qwen-Video-Edit.py) |
| Wan | [Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) | [code](/examples/wanvideo/model_inference/Wan2.1-T2V-1.3B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-T2V-1.3B.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-T2V-1.3B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-T2V-1.3B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B.py) |
| Wan | [Wan-AI/Wan2.1-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B) | [code](/examples/wanvideo/model_inference/Wan2.1-T2V-14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-T2V-14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-T2V-14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-T2V-14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-T2V-14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-14B.py) |
| Wan | [Wan-AI/Wan2.1-I2V-14B-480P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P) | [code](/examples/wanvideo/model_inference/Wan2.1-I2V-14B-480P.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-I2V-14B-480P.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-I2V-14B-480P.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-I2V-14B-480P.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-I2V-14B-480P.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-I2V-14B-480P.py) |
| Wan | [Wan-AI/Wan2.1-I2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P) | [code](/examples/wanvideo/model_inference/Wan2.1-I2V-14B-720P.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-I2V-14B-720P.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-I2V-14B-720P.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-I2V-14B-720P.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-I2V-14B-720P.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-I2V-14B-720P.py) |
| Wan | [Wan-AI/Wan2.1-FLF2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-FLF2V-14B-720P) | [code](/examples/wanvideo/model_inference/Wan2.1-FLF2V-14B-720P.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-FLF2V-14B-720P.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-FLF2V-14B-720P.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-FLF2V-14B-720P.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-FLF2V-14B-720P.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-FLF2V-14B-720P.py) |
| Wan | [iic/VACE-Wan2.1-1.3B-Preview](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview) | [code](/examples/wanvideo/model_inference/Wan2.1-VACE-1.3B-Preview.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-VACE-1.3B-Preview.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B-Preview.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B-Preview.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B-Preview.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-1.3B-Preview.py) |
| Wan | [Wan-AI/Wan2.1-VACE-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-1.3B) | [code](/examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-VACE-1.3B.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-1.3B.py) |
| Wan | [Wan-AI/Wan2.1-VACE-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B) | [code](/examples/wanvideo/model_inference/Wan2.1-VACE-14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-VACE-14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-VACE-14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-VACE-14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-14B.py) |
| Wan | [PAI/Wan2.1-Fun-1.3B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-1.3B-InP.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-1.3B-InP.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-1.3B-InP.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-1.3B-InP.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-1.3B-InP.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-1.3B-InP.py) |
| Wan | [PAI/Wan2.1-Fun-1.3B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-1.3B-Control.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-1.3B-Control.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-1.3B-Control.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-1.3B-Control.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-1.3B-Control.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-1.3B-Control.py) |
| Wan | [PAI/Wan2.1-Fun-14B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-InP) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-14B-InP.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-14B-InP.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-14B-InP.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-14B-InP.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-14B-InP.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-14B-InP.py) |
| Wan | [PAI/Wan2.1-Fun-14B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-Control) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-14B-Control.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-14B-Control.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-14B-Control.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-14B-Control.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-14B-Control.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-14B-Control.py) |
| Wan | [PAI/Wan2.1-Fun-V1.1-1.3B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-1.3B-Control.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-Control.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-Control.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-Control.py) |
| Wan | [PAI/Wan2.1-Fun-V1.1-14B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-14B-Control.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-Control.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-14B-Control.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-14B-Control.py) |
| Wan | [PAI/Wan2.1-Fun-V1.1-1.3B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-InP) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-InP.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-1.3B-InP.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-InP.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-InP.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-InP.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-InP.py) |
| Wan | [PAI/Wan2.1-Fun-V1.1-14B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-InP) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-InP.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-14B-InP.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-InP.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-14B-InP.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-InP.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-14B-InP.py) |
| Wan | [PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py) |
| Wan | [PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control-Camera.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-14B-Control-Camera.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-Control-Camera.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-14B-Control-Camera.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control-Camera.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-14B-Control-Camera.py) |
| Wan | [DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1) | [code](/examples/wanvideo/model_inference/Wan2.1-1.3b-speedcontrol-v1.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-1.3b-speedcontrol-v1.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-1.3b-speedcontrol-v1.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-1.3b-speedcontrol-v1.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-1.3b-speedcontrol-v1.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-1.3b-speedcontrol-v1.py) |
| Wan | [krea/krea-realtime-video](https://www.modelscope.cn/models/krea/krea-realtime-video) | [code](/examples/wanvideo/model_inference/krea-realtime-video.py) | [code](/examples/wanvideo/model_inference_low_vram/krea-realtime-video.py) | [code](/examples/wanvideo/model_training/full/krea-realtime-video.sh) | [code](/examples/wanvideo/model_training/validate_full/krea-realtime-video.py) | [code](/examples/wanvideo/model_training/lora/krea-realtime-video.sh) | [code](/examples/wanvideo/model_training/validate_lora/krea-realtime-video.py) |
| Wan | [meituan-longcat/LongCat-Video](https://www.modelscope.cn/models/meituan-longcat/LongCat-Video) | [code](/examples/wanvideo/model_inference/LongCat-Video.py) | [code](/examples/wanvideo/model_inference_low_vram/LongCat-Video.py) | [code](/examples/wanvideo/model_training/full/LongCat-Video.sh) | [code](/examples/wanvideo/model_training/validate_full/LongCat-Video.py) | [code](/examples/wanvideo/model_training/lora/LongCat-Video.sh) | [code](/examples/wanvideo/model_training/validate_lora/LongCat-Video.py) |
| Wan | [ByteDance/Video-As-Prompt-Wan2.1-14B](https://modelscope.cn/models/ByteDance/Video-As-Prompt-Wan2.1-14B) | [code](/examples/wanvideo/model_inference/Video-As-Prompt-Wan2.1-14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Video-As-Prompt-Wan2.1-14B.py) | [code](/examples/wanvideo/model_training/full/Video-As-Prompt-Wan2.1-14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Video-As-Prompt-Wan2.1-14B.py) | [code](/examples/wanvideo/model_training/lora/Video-As-Prompt-Wan2.1-14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Video-As-Prompt-Wan2.1-14B.py) |
| Wan | [Wan-AI/Wan2.2-T2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B) | [code](/examples/wanvideo/model_inference/Wan2.2-T2V-A14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-T2V-A14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-T2V-A14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-T2V-A14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-T2V-A14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-T2V-A14B.py) |
| Wan | [Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B) | [code](/examples/wanvideo/model_inference/Wan2.2-I2V-A14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-I2V-A14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-I2V-A14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-I2V-A14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-I2V-A14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-I2V-A14B.py) |
| Wan | [Wan-AI/Wan2.2-TI2V-5B](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B) | [code](/examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-TI2V-5B.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-TI2V-5B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-TI2V-5B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B.py) |
| Wan | [Wan-AI/Wan2.2-Animate-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B) | [code](/examples/wanvideo/model_inference/Wan2.2-Animate-14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-Animate-14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-Animate-14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-Animate-14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-Animate-14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-Animate-14B.py) |
| Wan | [Wan-AI/Wan2.2-Animate-2-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-2-14B) | [code](/examples/wanvideo/model_inference/Wan2.2-Animate-2-14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-Animate-2-14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-Animate-2-14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-Animate-2-14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-Animate-2-14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-Animate-2-14B.py) |
| Wan | [Wan-AI/Wan2.2-Animate-2-14B: Distilled](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-2-14B) | [code](/examples/wanvideo/model_inference/Wan2.2-Animate-2-14B-Distilled.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-Animate-2-14B-Distilled.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-Animate-2-14B-Distilled.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-Animate-2-14B-Distilled.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-Animate-2-14B-Distilled.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-Animate-2-14B-Distilled.py) |
| Wan | [Wan-AI/Wan2.2-S2V-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B) | [code](/examples/wanvideo/model_inference/Wan2.2-S2V-14B_multi_clips.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-S2V-14B_multi_clips.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-S2V-14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-S2V-14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-S2V-14B.py) |
| Wan | [PAI/Wan2.2-VACE-Fun-A14B](https://www.modelscope.cn/models/PAI/Wan2.2-VACE-Fun-A14B) | [code](/examples/wanvideo/model_inference/Wan2.2-VACE-Fun-A14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-VACE-Fun-A14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-VACE-Fun-A14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-VACE-Fun-A14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-VACE-Fun-A14B.py) |
| Wan | [PAI/Wan2.2-Fun-A14B-InP](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-InP) | [code](/examples/wanvideo/model_inference/Wan2.2-Fun-A14B-InP.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-Fun-A14B-InP.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-Fun-A14B-InP.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-Fun-A14B-InP.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-Fun-A14B-InP.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-Fun-A14B-InP.py) |
| Wan | [PAI/Wan2.2-Fun-A14B-Control](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control) | [code](/examples/wanvideo/model_inference/Wan2.2-Fun-A14B-Control.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-Fun-A14B-Control.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-Fun-A14B-Control.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-Fun-A14B-Control.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-Fun-A14B-Control.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-Fun-A14B-Control.py) |
| Wan | [PAI/Wan2.2-Fun-A14B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control-Camera) | [code](/examples/wanvideo/model_inference/Wan2.2-Fun-A14B-Control-Camera.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-Fun-A14B-Control-Camera.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-Fun-A14B-Control-Camera.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-Fun-A14B-Control-Camera.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-Fun-A14B-Control-Camera.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-Fun-A14B-Control-Camera.py) |
| Wan | [openmoss/MOVA-360p](https://modelscope.cn/models/openmoss/MOVA-360p) | [code](/examples/mova/model_inference/MOVA-360p-I2AV.py) | [code](/examples/mova/model_inference_low_vram/MOVA-360p-I2AV.py) | [code](/examples/mova/model_training/full/MOVA-360P-I2AV.sh) | [code](/examples/mova/model_training/validate_full/MOVA-360p-I2AV.py) | [code](/examples/mova/model_training/lora/MOVA-360P-I2AV.sh) | [code](/examples/mova/model_training/validate_lora/MOVA-360p-I2AV.py) |
| Wan | [openmoss/MOVA-720p](https://modelscope.cn/models/openmoss/MOVA-720p) | [code](/examples/mova/model_inference/MOVA-720p-I2AV.py) | [code](/examples/mova/model_inference_low_vram/MOVA-720p-I2AV.py) | [code](/examples/mova/model_training/full/MOVA-720P-I2AV.sh) | [code](/examples/mova/model_training/validate_full/MOVA-720p-I2AV.py) | [code](/examples/mova/model_training/lora/MOVA-720P-I2AV.sh) | [code](/examples/mova/model_training/validate_lora/MOVA-720p-I2AV.py) |
| Wan | [Wan-AI/Wan-Dancer-14B (global model)](https://modelscope.cn/models/Wan-AI/Wan-Dancer-14B) | [code](/examples/wanvideo/model_inference/Wan-Dancer-14B-global.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan-Dancer-14B-global.py) | [code](/examples/wanvideo/model_training/full/Wan-Dancer-14B-global.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan-Dancer-14B-global.py) | [code](/examples/wanvideo/model_training/lora/Wan-Dancer-14B-global.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan-Dancer-14B-global.py) |
| Wan | [Wan-AI/Wan-Dancer-14B (local model)](https://modelscope.cn/models/Wan-AI/Wan-Dancer-14B) | [code](/examples/wanvideo/model_inference/Wan-Dancer-14B-local.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan-Dancer-14B-local.py) | [code](/examples/wanvideo/model_training/full/Wan-Dancer-14B-local.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan-Dancer-14B-local.py) | [code](/examples/wanvideo/model_training/lora/Wan-Dancer-14B-local.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan-Dancer-14B-local.py) |
| FLUX.1 | [black-forest-labs/FLUX.1-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev) | [code](/examples/flux/model_inference/FLUX.1-dev.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev.py) | [code](/examples/flux/model_training/full/FLUX.1-dev.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev.py) | [code](/examples/flux/model_training/lora/FLUX.1-dev.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev.py) |
| FLUX.1 | [black-forest-labs/FLUX.1-Krea-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Krea-dev) | [code](/examples/flux/model_inference/FLUX.1-Krea-dev.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-Krea-dev.py) | [code](/examples/flux/model_training/full/FLUX.1-Krea-dev.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-Krea-dev.py) | [code](/examples/flux/model_training/lora/FLUX.1-Krea-dev.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-Krea-dev.py) |
| FLUX.1 | [black-forest-labs/FLUX.1-Kontext-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Kontext-dev) | [code](/examples/flux/model_inference/FLUX.1-Kontext-dev.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-Kontext-dev.py) | [code](/examples/flux/model_training/full/FLUX.1-Kontext-dev.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-Kontext-dev.py) | [code](/examples/flux/model_training/lora/FLUX.1-Kontext-dev.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-Kontext-dev.py) |
| FLUX.1 | [black-forest-labs/FLUX.1-Fill-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Fill-dev) | [code](/examples/flux/model_inference/FLUX.1-Fill-dev.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-Fill-dev.py) | [code](/examples/flux/model_training/full/FLUX.1-Fill-dev.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-Fill-dev.py) | [code](/examples/flux/model_training/lora/FLUX.1-Fill-dev.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-Fill-dev.py) |
| FLUX.1 | [black-forest-labs/FLUX.1-Redux-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Redux-dev) | [code](/examples/flux/model_inference/FLUX.1-Redux-dev.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-Redux-dev.py) | [code](/examples/flux/model_training/full/FLUX.1-Redux-dev.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-Redux-dev.py) | [code](/examples/flux/model_training/lora/FLUX.1-Redux-dev.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-Redux-dev.py) |
| FLUX.1 | [HuanJue/Insert-Anything](https://www.modelscope.cn/models/HuanJue/Insert-Anything) | [code](/examples/flux/model_inference/Insert-Anything.py) | [code](/examples/flux/model_inference_low_vram/Insert-Anything.py) | - | - | [code](/examples/flux/model_training/lora/Insert-Anything.sh) | [code](/examples/flux/model_training/validate_lora/Insert-Anything.py) |
| FLUX.1 | [alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta](https://www.modelscope.cn/models/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta) | [code](/examples/flux/model_inference/FLUX.1-dev-Controlnet-Inpainting-Beta.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-Controlnet-Inpainting-Beta.py) | [code](/examples/flux/model_training/full/FLUX.1-dev-Controlnet-Inpainting-Beta.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev-Controlnet-Inpainting-Beta.py) | [code](/examples/flux/model_training/lora/FLUX.1-dev-Controlnet-Inpainting-Beta.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev-Controlnet-Inpainting-Beta.py) |
| FLUX.1 | [InstantX/FLUX.1-dev-Controlnet-Union-alpha](https://www.modelscope.cn/models/InstantX/FLUX.1-dev-Controlnet-Union-alpha) | [code](/examples/flux/model_inference/FLUX.1-dev-Controlnet-Union-alpha.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-Controlnet-Union-alpha.py) | [code](/examples/flux/model_training/full/FLUX.1-dev-Controlnet-Union-alpha.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev-Controlnet-Union-alpha.py) | [code](/examples/flux/model_training/lora/FLUX.1-dev-Controlnet-Union-alpha.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev-Controlnet-Union-alpha.py) |
| FLUX.1 | [jasperai/Flux.1-dev-Controlnet-Upscaler](https://www.modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Upscaler) | [code](/examples/flux/model_inference/FLUX.1-dev-Controlnet-Upscaler.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-Controlnet-Upscaler.py) | [code](/examples/flux/model_training/full/FLUX.1-dev-Controlnet-Upscaler.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev-Controlnet-Upscaler.py) | [code](/examples/flux/model_training/lora/FLUX.1-dev-Controlnet-Upscaler.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev-Controlnet-Upscaler.py) |
| FLUX.1 | [InstantX/FLUX.1-dev-IP-Adapter](https://www.modelscope.cn/models/InstantX/FLUX.1-dev-IP-Adapter) | [code](/examples/flux/model_inference/FLUX.1-dev-IP-Adapter.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-IP-Adapter.py) | [code](/examples/flux/model_training/full/FLUX.1-dev-IP-Adapter.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev-IP-Adapter.py) | [code](/examples/flux/model_training/lora/FLUX.1-dev-IP-Adapter.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev-IP-Adapter.py) |
| FLUX.1 | [ByteDance/InfiniteYou](https://www.modelscope.cn/models/ByteDance/InfiniteYou) | [code](/examples/flux/model_inference/FLUX.1-dev-InfiniteYou.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-InfiniteYou.py) | [code](/examples/flux/model_training/full/FLUX.1-dev-InfiniteYou.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev-InfiniteYou.py) | [code](/examples/flux/model_training/lora/FLUX.1-dev-InfiniteYou.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev-InfiniteYou.py) |
| FLUX.1 | [DiffSynth-Studio/Eligen](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen) | [code](/examples/flux/model_inference/FLUX.1-dev-EliGen.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-EliGen.py) | - | - | [code](/examples/flux/model_training/lora/FLUX.1-dev-EliGen.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev-EliGen.py) |
| FLUX.1 | [DiffSynth-Studio/LoRA-Encoder-FLUX.1-Dev](https://www.modelscope.cn/models/DiffSynth-Studio/LoRA-Encoder-FLUX.1-Dev) | [code](/examples/flux/model_inference/FLUX.1-dev-LoRA-Encoder.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-LoRA-Encoder.py) | [code](/examples/flux/model_training/full/FLUX.1-dev-LoRA-Encoder.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev-LoRA-Encoder.py) | - | - |
| FLUX.1 | [DiffSynth-Studio/LoRAFusion-preview-FLUX.1-dev](https://modelscope.cn/models/DiffSynth-Studio/LoRAFusion-preview-FLUX.1-dev) | [code](/examples/flux/model_inference/FLUX.1-dev-LoRA-Fusion.py) | - | - | - | - | - |
| FLUX.1 | [stepfun-ai/Step1X-Edit](https://www.modelscope.cn/models/stepfun-ai/Step1X-Edit) | [code](/examples/flux/model_inference/Step1X-Edit.py) | [code](/examples/flux/model_inference_low_vram/Step1X-Edit.py) | [code](/examples/flux/model_training/full/Step1X-Edit.sh) | [code](/examples/flux/model_training/validate_full/Step1X-Edit.py) | [code](/examples/flux/model_training/lora/Step1X-Edit.sh) | [code](/examples/flux/model_training/validate_lora/Step1X-Edit.py) |
| FLUX.1 | [ostris/Flex.2-preview](https://www.modelscope.cn/models/ostris/Flex.2-preview) | [code](/examples/flux/model_inference/FLEX.2-preview.py) | [code](/examples/flux/model_inference_low_vram/FLEX.2-preview.py) | [code](/examples/flux/model_training/full/FLEX.2-preview.sh) | [code](/examples/flux/model_training/validate_full/FLEX.2-preview.py) | [code](/examples/flux/model_training/lora/FLEX.2-preview.sh) | [code](/examples/flux/model_training/validate_lora/FLEX.2-preview.py) |
| FLUX.1 | [DiffSynth-Studio/Nexus-GenV2](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2) | [code](/examples/flux/model_inference/Nexus-Gen-Editing.py) | [code](/examples/flux/model_inference_low_vram/Nexus-Gen-Editing.py) | [code](/examples/flux/model_training/full/Nexus-Gen.sh) | [code](/examples/flux/model_training/validate_full/Nexus-Gen.py) | [code](/examples/flux/model_training/lora/Nexus-Gen.sh) | [code](/examples/flux/model_training/validate_lora/Nexus-Gen.py) |
| Stable Diffusion XL | [stabilityai/stable-diffusion-xl-base-1.0](https://www.modelscope.cn/models/stabilityai/stable-diffusion-xl-base-1.0) | [code](/examples/stable_diffusion_xl/model_inference/stable-diffusion-xl-base-1.0.py) | [code](/examples/stable_diffusion_xl/model_inference_low_vram/stable-diffusion-xl-base-1.0.py) | [code](/examples/stable_diffusion_xl/model_training/full/stable-diffusion-xl-base-1.0.sh) | [code](/examples/stable_diffusion_xl/model_training/validate_full/stable-diffusion-xl-base-1.0.py) | [code](/examples/stable_diffusion_xl/model_training/lora/stable-diffusion-xl-base-1.0.sh) | [code](/examples/stable_diffusion_xl/model_training/validate_lora/stable-diffusion-xl-base-1.0.py) |
| Stable Diffusion | [AI-ModelScope/stable-diffusion-v1-5](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5) | [code](/examples/stable_diffusion/model_inference/stable-diffusion-v1-5.py) | [code](/examples/stable_diffusion/model_inference_low_vram/stable-diffusion-v1-5.py) | [code](/examples/stable_diffusion/model_training/full/stable-diffusion-v1-5.sh) | [code](/examples/stable_diffusion/model_training/validate_full/stable-diffusion-v1-5.py) | [code](/examples/stable_diffusion/model_training/lora/stable-diffusion-v1-5.sh) | [code](/examples/stable_diffusion/model_training/validate_lora/stable-diffusion-v1-5.py) |
| - | [PickScore](https://github.com/yuvalkirstain/pickscore) | [code](/examples/image_quality_metric/pickscore.py) | - | - | - | - | - |
| - | [ImageReward](https://github.com/zai-org/ImageReward) | [code](/examples/image_quality_metric/image_reward.py) | - | - | - | - | - |
| - | [HPSv2](https://github.com/tgxs002/HPSv2) | [code](/examples/image_quality_metric/hpsv2.py) | - | - | - | - | - |
| - | [HPSv3](https://github.com/MizzenAI/HPSv3) | [code](/examples/image_quality_metric/hpsv3.py) | - | - | - | - | - |
| - | [CLIP Score](https://github.com/openai/CLIP) | [code](/examples/image_quality_metric/clipscore.py) | - | - | - | - | - |
| - | [UnifiedReward 2.0](https://github.com/THUDM/UnifiedReward) | [code](/examples/image_quality_metric/unified_reward_2.py) | - | - | - | - | - |
| - | [Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) | [code](/examples/image_quality_metric/qwen_image_bench.py) | - | - | - | - | - |
| - | [UnifiedReward Edit](https://github.com/THUDM/UnifiedReward) | [code](/examples/image_quality_metric/unified_reward_edit.py) | - | - | - | - | - |
| - | [Aesthetic](https://github.com/christophschuhmann/improved-aesthetic-predictor) | [code](/examples/image_quality_metric/aesthetic.py) | - | - | - | - | - |
| - | [FID](https://github.com/mseitzer/pytorch-fid) | [code](/examples/image_quality_metric/fid.py) | - | - | - | - | - |
