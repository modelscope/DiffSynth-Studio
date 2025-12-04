# `diffsynth.core.data`: 数据处理算子与通用数据集

## 数据处理算子

### 可用数据处理算子

`diffsynth.core.data` 提供了一系列数据处理算子，用于进行数据处理，包括：

* 数据格式转换算子
    * `ToInt`: 转换为 int 格式
    * `ToFloat`: 转换为 float 格式
    * `ToStr`: 转换为 str 格式
    * `ToList`: 转换为列表格式，以列表包裹此数据
    * `ToAbsolutePath`: 将相对路径转换为绝对路径
* 文件加载算子
    * `LoadImage`: 读取图片文件
    * `LoadVideo`: 读取视频文件
    * `LoadAudio`: 读取音频文件
    * `LoadGIF`: 读取 GIF 文件
    * `LoadTorchPickle`: 读取由 [`torch.save`](https://docs.pytorch.org/docs/stable/generated/torch.save.html) 保存的二进制文件【该算子可能导致二进制文件中的代码注入攻击，请谨慎使用！】
* 媒体文件处理算子
    * `ImageCropAndResize`: 对图像进行裁剪和拉伸
* Meta 算子
    * `SequencialProcess`: 将序列中的每个数据路由到一个算子
    * `RouteByExtensionName`: 按照文件扩展名路由到特定算子
    * `RouteByType`: 按照数据类型路由到特定算子

### 算子使用

数据算子之间以 `>>` 符号连接形成数据处理流水线，例如：

```python
from diffsynth.core.data.operators import *

data = "image.jpg"
data_pipeline = ToAbsolutePath(base_path="/data") >> LoadImage() >> ImageCropAndResize(max_pixels=512*512)
data = data_pipeline(data)
```

在经过每个算子后，数据被依次处理

* `ToAbsolutePath(base_path="/data")`: `"/data/image.jpg"`
* `LoadImage()`: `<PIL.Image.Image image mode=RGB size=1024x1024 at 0x7F8E7AAEFC10>`
* `ImageCropAndResize(max_pixels=512*512)`: `<PIL.Image.Image image mode=RGB size=512x512 at 0x7F8E7A936F20>`

我们可以组合出功能完备的数据流水线，例如通用数据集的默认视频数据算子为

```python
RouteByType(operator_map=[
    (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
        (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
        (("gif",), LoadGIF(
            num_frames, time_division_factor, time_division_remainder,
            frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
        )),
        (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
            num_frames, time_division_factor, time_division_remainder,
            frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
        )),
    ])),
])
```

它包含如下逻辑：

* 如果是 `str` 类型的数据
    * 如果是 `"jpg", "jpeg", "png", "webp"` 类型文件
        * 加载这张图片
        * 裁剪并缩放到特定分辨率
        * 打包进列表，视为单帧视频
    * 如果是 `"gif"` 类型文件
        * 加载 gif 文件内容
        * 将每一帧裁剪和缩放到特定分辨率
    * 如果是 `"mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"` 类型文件
        * 加载 gif 文件内容
        * 将每一帧裁剪和缩放到特定分辨率
* 如果不是 `str` 类型的数据，报错

## 通用数据集

`diffsynth.core.data` 提供了统一的数据集实现，数据集需输入以下参数：

* `base_path`: 根目录，若数据集中包含图片文件的相对路径，则需填入此字段用于加载这些路径指向的文件
* `metadata_path`: 元数据目录，记录所有元数据的文件路径，支持 `csv`、`json`、`jsonl` 格式
* `repeat`: 数据重复次数，默认为 1，该参数影响一个 epoch 的训练步数
* `data_file_keys`: 需进行加载的数据字段名，例如 `(image, edit_image)`
* `main_data_operator`: 主加载算子，需通过数据处理算子组装好数据处理流水线
* `special_operator_map`: 特殊算子映射，对需要特殊处理的字段构建的算子映射

### 元数据

数据集的 `metadata_path` 指向元数据文件，支持 `csv`、`json`、`jsonl` 格式，以下提供了样例

* `csv` 格式：可读性高、不支持列表数据、内存占用小

```csv
image,prompt
image_1.jpg,"a dog"
image_2.jpg,"a cat"
```

* `json` 格式：可读性高、支持列表数据、内存占用大

```json
[
    {
        "image": "image_1.jpg",
        "prompt": "a dog"
    },
    {
        "image": "image_2.jpg",
        "prompt": "a cat"
    }
]
```

* `jsonl` 格式：可读性低、支持列表数据、内存占用小

```json
{"image": "image_1.jpg", "prompt": "a dog"}
{"image": "image_2.jpg", "prompt": "a cat"}
```

如何选择最佳的元数据格式？

* 如果数据量大，达到千万级的数据量，由于 `json` 文件解析时需要额外内存，此时不可用，请使用 `csv` 或 `jsonl` 格式
* 如果数据集中包含列表数据，例如编辑模型需输入多张图，由于 `csv` 格式无法存储列表格式数据，此时不可用，请使用 `json` 或 `jsonl` 格式

### 数据加载逻辑

在没有进行额外设置时，数据集默认输出元数据集中的数据，图片和视频文件的路径会以字符串的格式输出，若要加载这些文件，则需要设置 `data_file_keys`、`main_data_operator`、`special_operator_map`。

在数据处理流程中，按如下逻辑进行处理：
* 如果字段位于 `special_operator_map`，则调用 `special_operator_map` 中的对应算子进行处理
* 如果字段不位于 `special_operator_map`
    * 如果字段位于 `data_file_keys`，则调用 `main_data_operator` 算子进行处理
    * 如果字段不位于 `data_file_keys`，则不进行处理

`special_operator_map` 可用于实现特殊的数据处理，例如模型 [Wan-AI/Wan2.2-Animate-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B) 中输入的人物面部视频 `animate_face_video` 是以固定分辨率处理的，与输出视频不一致，因此这一字段由专门的算子处理：

```python
special_operator_map={
    "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
}
```

### 其他注意事项

当数据量过少时，可适当增加 `repeat`，延长单个 epoch 的训练时间，避免频繁保存模型产生较多耗时。

当数据量 * `repeat` 超过 $10^9$ 时，我们观测到数据集的速度明显变慢，这似乎是 `PyTorch` 的 bug，我们尚不确定新版本的 `PyTorch` 是否已经修复了这一问题。
