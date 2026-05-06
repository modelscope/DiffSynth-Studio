# 推理 WebUI

DiffSynth-Studio 提供推理 WebUI，帮助开发者快速验证模型效果。

> 推理 WebUI 是面向开发者的调试工具，而非面向创作者的设计工具。如需功能更丰富、交互更友好的创作体验，推荐使用魔搭社区 [AIGC 专区](https://modelscope.cn/aigc/home)（中国用户）或 [Civision 专区](https://modelscope.ai/civision/home)（非中国用户）。

## 启动推理 WebUI

推理 WebUI 基于 [Streamlit](https://streamlit.io/) 构建。建议以 `[all]` 模式安装 DiffSynth-Studio：

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .[all]
```

启动命令：

```shell
streamlit run examples/dev_tools/webui.py --server.fileWatcherType none
```

## 工作原理

推理 WebUI 作为独立工具，通过解析 Pipeline 的 `from_pretrained` 和 `__call__` 方法中的参数类型标注，动态生成对应 UI 控件。因此，界面交互逻辑与代码调用逻辑完全一致，可视为 DiffSynth-Studio 代码的可视化入口。

以 `diffsynth.pipelines.z_image` 中的 `ZImagePipeline.__call__` 为例：

```python
@torch.no_grad()
def __call__(
    self,
    # Prompt
    prompt: str = "",
    negative_prompt: str = "",
    cfg_scale: float = 1.0,
    # Image
    input_image: Image.Image = None,
    denoising_strength: float = 1.0,
    ...
)
```

WebUI 解析后将自动渲染为如下界面：

![](https://github.com/user-attachments/assets/55795022-7a9b-4383-b048-7feabdfcdddf)

## 使用提示

- 支持从 `./examples` 样例代码中自动加载 `model_id`、`origin_file_pattern` 等模型信息，简化配置流程；
- `vram_limit`、`tokenizer_config`、`lora` 等参数无法通过代码解析获取，需手动填写。
