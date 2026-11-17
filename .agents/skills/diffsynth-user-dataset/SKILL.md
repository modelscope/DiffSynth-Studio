---
name: diffsynth-user-dataset
description: >
  Prepare, run, troubleshoot or resume standard LoRA and
  full fine-tuning of supported DiffSynth-Studio models on
  user datasets. Includes dataset preparation, training
  memory optimization, distributed training and checkpoint
  validation. Not for new training algorithms or research
  comparisons; use diffsynth-user-research for those.
  标准 LoRA/全量微调、训练资源配置及权重验证。
---

# DiffSynth-Studio 用户训练数据集准备

以用户正在使用的源码和对应 `model_training` 示例为准，确定字段如何被加载和消费。保留原始数据，转换结果写到新位置，不默认下载模型或启动训练。

请求同时包含标准全量/LoRA 训练时，由同级 [diffsynth-user-training](../diffsynth-user-training/SKILL.md) 主导模型、资源与训练配置，本 Skill 完成数据验收和交接后继续原训练任务；不能仅交付数据就结束整个请求。仅要求数据格式、检查或转换时直接完成本 Skill，不顺带执行训练。

先匹配请求范围，再执行下面相关步骤：

- **解释格式**：给出目标入口的字段、类型、真实记录示例和参数，不额外转换数据。
- **检查已有数据**：检查用户指定的数据，报告问题与验证范围；只要求检查时不生成修复版。
- **准备可训练数据**：实际生成 metadata 和必要媒体转换，并完成真实 Dataset 抽样验收。环境或必要标注不足时交付已完成部分，明确未验证项，不宣称可直接训练。

编辑、视频、音频及 Template 的特殊字段按实际处理代码核对。优先利用现有 Python/CLI；通用检查使用 [scripts/check_dataset.py](scripts/check_dataset.py)，任务特有转换按需要写在用户项目中。

## 步骤 1：确定目标训练入口和原始数据

从用户目标、项目及现有文件确定：模型/变体、训练任务（SFT/LoRA/编辑/控制/视频/音频/Template 等）、所用 DiffSynth-Studio 源码根目录、原始数据位置、现有标注/配对关系、希望输出的位置、是否允许媒体转码或自动标注。通用格式咨询可直接解释共性与入口差异；只有具体转换或兼容性判断依赖目标入口、且无法从现有上下文确定时才询问必要信息，不自动替换为另一模型。

先抽查目录、少量真实标注和媒体元信息，了解来源格式、大小、缺失字段、重复/损坏文件及对应关系。不要仅按文件名推断源图与目标图、视频与音频的配对；不擅自编造 caption 或改写监督目标。

## 步骤 2：从官方训练示例反推数据契约

在**实际源码版本**中，优先读取匹配的：

1. `examples/<family>/model_training/{lora,full,special}/<variant>.sh`：确认 `--dataset_base_path`、`--dataset_metadata_path`、`--data_file_keys`、`--extra_inputs`、`--num_frames` 等真实参数，以及 `--task` 是否为拆分训练。
2. `examples/<family>/model_training/train.py`：核对 `UnifiedDataset` 构造、`main_data_operator`、`special_operator_map`、`get_pipeline_inputs()`、模型专有的 `parse_extra_inputs()`。
3. `diffsynth/core/data/{unified_dataset,operators}.py` 和使用到的专用 processor / Template `TEMPLATE_DATA_PROCESSOR`；必要时核对 Pipeline unit 实际需要的条件字段。

在执行转换前，确定一份**任务内的字段对应关系**（直接在工作记录中即可，不强制创建 schema 文件）：每个目标字段的来源、数据类型、是否必需、是否位于 `data_file_keys`、使用哪个 operator、是否通过 `extra_inputs` 或 `get_pipeline_inputs()` 进入训练。检查三段链路：metadata 有值 → operator 实际加载为正确类型 → 训练模块实际消费。`special_operator_map` 仅对位于 `data_file_keys` 中的字段生效；`--extra_inputs` 不自动加载文件。

**不要把 `extra_inputs` 全部当作元数据列。** 例如 Wan I2V 的 `input_image` 通常从目标 `video[0]` 派生，MiniMax-H3 的 `input_image/end_image` 从视频首末帧派生；但 Qwen-Image-Edit 的 `edit_image` 是独立条件，必须进入相应的文件加载键。字段名相似不代表监督方向相同。

## 步骤 3：设计最小输出结构与元数据格式

按照真实 loader 选择 CSV、JSON 或 JSONL：

- 单值字符串/数值字段可用 CSV；若包含多图列表、嵌套 reference、Template 输入，优先 JSONL（或 JSON 顶层数组），**保留原生 JSON 类型**。不要把 `['a.png']` 的 CSV 字符串假装成可被默认图像 operator 解析的列表。
- 新元数据建议使用小写 `.csv`、`.json` 或 `.jsonl` 扩展名。框架仅精确匹配小写 JSON 后缀，其他后缀进入 CSV 分支；本检查器支持大小写不限的 `.csv` 及小写 `.json/.jsonl`，不尝试猜测其他扩展名的格式。当前 `UnifiedDataset` 的 JSONL **逐行 `strip()` 后 `json.loads`，空行会失败**；JSON/JSONL 不写 UTF-8 BOM。它会将所有记录保存到内存，不是训练时流式 Dataset。CSV 由 pandas 读取，空值和 `NA` 等字符串可能变成 NaN，纯数字样式的路径或 ID 还可能被推断成数值、丢失前导零；优先将这类数据输出为 JSONL，并真实加载验证。
- 经 `ToAbsolutePath` 的顶层路径按 `--dataset_base_path` 解析，与 metadata 所在目录不一定相同，也不会自动展开 `~`。**嵌套 Template 路径由专用 processor 决定**；例如 brightness 的 `DataAnnotator` 直接 `Image.open(image)`，相对路径基于进程工作目录，不会自动拼接数据根目录。明确训练命令的工作目录及各类路径的解析规则。
- 优先保留无需改动的原始媒体、仅生成新 metadata；只有必要的转码、裁剪、片段提取、重采样等才产生新媒体，全部写到新目录。使用相对路径时验证它们从约定根目录能够解析；确需绝对路径应注明不可移植性。
- 保留稳定 `sample_id` 或可追溯来源索引。需要筛选、重新配对或重命名时保留原记录到新记录的对应关系；不直接覆盖来源文件。

`metadata_path=None` 在该 loader 中表示递归读取 `.pth` 训练缓存，**不是**自动扫描原始图像/视频目录。拆分训练的 `sft:data_process` 阶段读取原始 metadata；`sft:train` 阶段可读取可信的预处理缓存。另一个不同用途是 DiffSynth-Music 的 metadata 中 `audio` 指向 `.pth` 音频 latent，由 `LoadMultiTrackAudio` 加载为 Tensor；不能仅凭扩展名把它当作整条训练记录的拆分缓存。不要替用户手工伪造缓存；不加载来源不明的 PyTorch pickle。

## 步骤 4：真正执行数据转换与标注

用户要求准备或转换数据时，按步骤 2 的字段映射生成真实 CSV/JSON/JSONL：规范化列名与数据类型、修复可确认的配对、构造正确的列表/嵌套结构、处理必要的文件路径和媒体格式。不要改变用户指定的监督语义。

- 图像：记录 RGB/RGBA、alpha、EXIF 方向、损坏和极端长宽比；需要转码或预裁剪时先评估是否会破坏目标。最终空间尺寸由目标训练 operator 的 `max_pixels`、`height/width` 及 division factor 决定，不盲目预先统一缩放。
- 编辑/控制：确定**监督目标**与**条件输入**各是哪列；多图条件保持顺序与数量；图像编辑不能把 source 和 target 反过来。图层任务可要求 RGBA 或专门字段。
- 视频：检查时长、真实可读帧数、fps、片段起止和音视频关系。目标 `num_frames`、`time_division_factor/remainder`、`fix_frame_rate` 以训练脚本为准；默认视频 operator 接收的是**文件路径字符串**，不是任意帧路径列表。视频不足、取帧后变短或形状不合要求时，不把预期帧数当作实际输出。
- LTX2/MOVA 的帧率来源需额外核对：当前 Dataset 的视频/音频采样使用 `args.frame_rate`，`get_pipeline_inputs()` 却读取 `data.get("frame_rate", 24)`。使用非默认帧率时，在每条 metadata 中填入与 CLI 一致的数值型 `frame_rate`；只修改 `--frame_rate` 不会同步改变 Pipeline 的默认 24fps。核对采样后的帧数、音频时长与最终输入值一致。
- 音频：检查声道、时长、可解码性与数据角色；是否重采样、截断、补静音，以目标 audio operator 和用户意图为准。某些 operator 失败时返回 `None` 而不抛异常，验收时必须检查值。多轨音频或音视频联合条件要核对结构和对齐。
- Caption/lyrics/指令：优先使用用户已有标注。只有用户授权的模型/服务才能自动生成或改写；保留原文、生成状态与失败样本，不把占位文本当成真实完成的 caption。
- 训练/验证划分只在需要时生成；按主体、原始视频/片段、重复或近重复组隔离，避免同源泄漏。默认训练入口通常只接受一个 `dataset_metadata_path`，不能假定额外的验证文件会被它自动使用。`dataset_repeat` 用训练参数控制，不物理复制记录。

对于 Qwen-Image-Edit，输出记录形如 `{"sample_id":"001","image":"targets/001.png","prompt":"把杯子变成蓝色","edit_image":["sources/001.png"]}`；`image` 是目标而不是来源。复杂 reference 则按 MiniMax-H3 的 `{"type":"image","image":"refs/001.png"}` 等原生对象形式输出，不将对象序列化成 CSV 字符串。

## 步骤 5：结构与媒体预检

使用通用脚本检查**路径字符串/列表、文本类型、文件存在性和基础媒体可读性**。例如 Qwen-Image-Edit：

```bash
python /path/to/diffsynth-user-dataset/scripts/check_dataset.py \
  /path/to/output/metadata.jsonl --base /path/to/dataset_root \
  --require image,prompt,edit_image --file-keys image,edit_image \
  --text-keys prompt --decode --output /path/to/output/dataset-check.json
```

`--require` 指定必需且非空的字段；`--file-keys` 指定路径字段，未出现的列默认可选。只有目标 operator 明确允许空条件时，才用 `--nullable-file-keys context_image` 允许该字段为 null/CSV NaN；它必须位于 `--file-keys` 且不能同时位于 `--require`。有稳定 ID 时增加 `--id-key sample_id` 检查缺失和重复。

空列表与 null 是不同的契约。例如 H3 的 `MiniMaxH3ReferenceLoader` 接受 `references: []`，ReferenceEncoder 会跳过参考编码；此时可加 `--allow-empty-file-keys references`。该参数只允许显式声明字段的顶层空列表，同样必须位于 `--file-keys` 且不能同时位于 `--require`。是否允许空列表由目标 operator 和消费者共同决定，不对所有图像或条件字段统一放宽。

`--text-keys` 指定本任务**只能为字符串**的字段，默认 `prompt,lyrics` 是常规配方的便捷配置，不是全框架类型约束。空字符串是否允许由任务决定，需要非空时同时列入 `--require`。例如 LingBot 的 `prompt` 可为原生字典，先核对 `LingBotVideoUnit_PromptEmbedder.normalize_caption`；检查时用 `--text-keys lyrics` 排除 prompt 的字符串断言，或在没有字符串字段时用 `--text-keys ""`，并单独验证结构化字段的 processor。`--require prompt` 仍可检查该字段非空；不能用关闭字符串检查代替目标入口的类型验收。

不加 `--decode` 时只检查结构、类型和路径；加上后图片按内容识别并执行完整像素读取（含动图各帧，例如 `.jfif` 也可识别），视频/音频用 ffprobe 检查媒体流与元信息，**不代表视频逐帧解码或音频语义正确**。脚本中的扩展名集合和字段名提示用于选择探测与报告方式，不是框架支持格式的白名单；可读的文件仍须通过目标 operator 的类型与扩展名路由。`.pth/.pt/.safetensors/.npy/.npz` 等序列化输入只检查文件存在性，不反序列化，启用 `--decode` 时列为待目标 operator 验证；其他无法识别的格式也不能仅据通用探测失败就断言是坏样本。

图片解码需要 Pillow，CSV 检查需要目标训练环境中的 pandas：脚本先做额外的表头唯一性与行宽质量检查，再用 `pandas.read_csv` 检查解析后的值，不安装依赖。CSV 的未引用纯空格/制表符行按 pandas 行为跳过，带引号的空白字段与多行 caption 保留原内容。脚本还拒绝 JSON 中非标准的 `NaN/Infinity` 常量；这是工具的质量约束，框架默认 JSON 解析器本身可能接受这些常量。额外质量检查通过不保证 loader 的全部行为一致；结构解析失败会终止本次检查，修复后重跑，不把中止前的记录当作全量验收。

报告使用**尚不存在的新文件路径**，脚本拒绝覆盖已有文件（包括通过链接指向的原始数据）；这是本工具的写入保护约定，重跑时选择新报告名，或省略 `--output` 输出到终端。`static_checked` 仅表示这些预检通过，不代表可直接训练；`needs_loader_check` 表示尚有 URL、latent 或未知格式等待验项；`failed` 表示存在检查错误。正常检查的退出码 0 表示预检通过，1 表示失败或存在待验项，CLI 参数错误由 argparse 返回 2；自动化时同时读取 JSON 状态，不能把退出码 1 一律解释成数据损坏。报告分别记录错误与 `pending_checks`，`--max-errors` 限制两类详情各自的条数，总数和截断数仍保留。待验项经真实 operator 验证后另行记录结果，不因通用工具尚不支持就排除样本。

对于 `references` 等特殊结构，脚本只能提取路径，仍需使用真实专用 processor 验证；路径字典被预检接受不表示默认图像 operator 接受字典。Template 不是自动递归加载的普通路径字段，不直接把整个 `template_inputs` 填进 `--file-keys`。URL 图片虽然可能被官方 loader 支持，但 `--allow-urls` 仅标记未验证，不发起网络请求。

按行号报告问题；用户要求修复时，将有依据的修改写入新数据。检查缺失/错误路径、不符合任务要求的空 caption、CSV 列表字符串、重复 ID、错误配对及损坏媒体。无可修复依据的样本保留排除原因，不静默删除。

## 步骤 6：真实 UnifiedDataset 抽样验收

交付“可直接训练”的数据前，用**目标训练环境及 `train.py` 中相同的 Dataset 构造参数、main_data_operator 和 special_operator_map**实例化 `UnifiedDataset`，读取头部、中间、尾部及随机样本。只构建 Dataset，不执行 `train.py` 的主程序或训练模块构造器，以免加载权重。抽样数量按数据规模与风险决定，包含复杂结构、异常修复和各媒体类型；抽样不等于全量媒体均已验证。用户只要求静态检查或环境不可用时，如实标记未执行此步。

检查加载后的实际 Python 类型与内容：普通文本字段应为字符串且没有意外 NaN，结构化 caption 按目标 processor 接受的类型检查，不统一转成字符串；单图是 PIL Image，多图条件是有序列表，视频是按 operator 采样后的帧列表。音频按实际算子核对：`LoadAudio` 返回 waveform 数组，Torchaudio 系列算子通常返回 `(waveform, sample_rate)`，DiffSynth-Music 的 `LoadMultiTrackAudio` 在当前字符串路径入口返回 waveform/latent Tensor。验证真实帧数、尺寸、采样率、时间对齐和必需条件非 `None`，可选空条件按配方判断；尤其注意音频算子可能吞掉解码异常并返回 `None`。Template 输入可能仍保留原始嵌套对象，需要另外核对其 processor，不把 Dataset 返回成功当作 processor 已运行。

对照 `get_pipeline_inputs()`、`parse_extra_inputs()` 或 Template processor，确认字段进入相应输入分支。能在不下载/加载模型情况下检查的输入映射直接核查；需要模型的部分保留“待训练链路验证”，**不把只读 Dataset 成功说成已运行训练或模型已消费条件**。仅准备数据时不运行权重加载、前向或训练；当前请求已包含训练时，验收后直接交接已验证的数据和参数，无需用户重复请求。

## 步骤 7：交付真正可训练的数据集

交付实际生成的 metadata 文件及必要媒体目录、准确的 `--dataset_base_path`、`--dataset_metadata_path`、`--data_file_keys`、`--extra_inputs` 和模型专属参数。提供基于**匹配官方 shell**改写的训练入口命令片段（不默认执行），并说明训练用的工作目录。

简要报告：原始/有效/排除数量、文件与媒体检查结果、标注状态、可选划分与来源映射、UnifiedDataset 抽样结果、需要人工确认的质量或配对问题。字段未通过真实 operator 或存在未处理的必要标注时明确标注**尚未达到可直接训练**，不得报告“已准备完成”。不要产生与交付无关的数据库、缓存、模型权重、样例训练日志或冗余代码。

## 常见训练入口差异（导航，使用时仍以当前源码为准）

| 任务/入口 | 重点字段与注意事项 |
| --- | --- |
| Z-Image / 常规文生图 | `image` 是监督目标，`prompt` 是文本；通常 `--data_file_keys image` |
| Qwen-Image-Edit | `image` 是目标，`edit_image` 是输入图（可为列表）；`--data_file_keys image,edit_image`，`--extra_inputs edit_image`；Layered 另查 RGBA/图层字段 |
| Qwen-Video-Edit | `video` 是目标，`input_video` 是编辑条件；两个字段均须作为媒体加载 |
| Wan I2V | 视频为监督目标；`--extra_inputs input_image` 常由 `video[0]` 派生，不必额外编造一列 |
| LingBot-Video | `video` 为目标；`prompt` 可为字符串或原生字典，核对 PromptEmbedder 的 caption 规范化逻辑；字典 prompt 不列入 `--text-keys`，首帧条件按目标配方处理 |
| MiniMax-H3 FL2VA/Ref2VA | `video`、可能的 `input_audio`、Ref2VA 的嵌套 `references`；参照专用 `MiniMaxH3ReferenceLoader`，可选无参考样本可用 `references: []`；首尾帧由视频派生，核对 17n+5 分组 |
| LTX2 | Dataset 配置 `time_division_factor/remainder=8/1`、空间 division factor=32；实际帧数仍须验收。`input_audio` 与 `in_context_videos` 使用专用 operator，后者由 Pipeline 的 InContextVideoEmbedder 消费；核对 CLI、metadata 与 Pipeline 的帧率一致 |
| MOVA | Dataset 配置 `time_division_factor/remainder=4/1`、空间 division factor=16；实际帧数仍须验收。`input_audio` 使用专用 operator，`input_image` 可由目标视频首帧派生；当前 train.py 虽注册 `in_context_videos` loader，Pipeline units 与 model_fn 未消费该字段，不能据注册声称支持该条件；核对帧率来源 |
| ACE-Step 1.5 | `audio` 经目标 operator 加载，`prompt`、`lyrics` 为文本；`bpm`、`keyscale` 等按实际训练代码使用 |
| DiffSynth-Music | `prompt`、`lyrics` 为文本；当前 `train.py` 的 `audio` 经 `ToAbsolutePath >> LoadMultiTrackAudio`，接收原始音频或 `.pth` latent 的字符串路径，不能直接填入多轨字典；latent 形状按 InputAudioEmbedder 核验，不用 ffprobe 判断有效性 |
| Template | 核对入口是否支持 Template、`--extra_inputs template_inputs` 和所选 `TEMPLATE_DATA_PROCESSOR`；嵌套字段的类型、路径解析及处理时机由该 processor 决定 |

权威来源：`diffsynth/core/data/unified_dataset.py`、`diffsynth/core/data/operators.py`、`docs/zh/API_Reference/core/data.md`、`docs/zh/Training/Supervised_Fine_Tuning.md`、`docs/zh/Diffusion_Templates/Template_Model_Training.md` 和目标 `examples/<family>/model_training/` 的 shell/`train.py`。源码与文档有差异时，以用户实际要运行的代码和真实 loader 行为为准。
