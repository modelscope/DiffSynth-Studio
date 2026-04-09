# `diffsynth.core.data`: Data Processing Operators and Universal Dataset

## Data Processing Operators

### Available Data Processing Operators

`diffsynth.core.data` provides a series of data processing operators for data processing, including:

* Data format conversion operators
    * `ToInt`: Convert to int format
    * `ToFloat`: Convert to float format
    * `ToStr`: Convert to str format
    * `ToList`: Convert to list format, wrapping this data in a list
    * `ToAbsolutePath`: Convert relative paths to absolute paths
* File loading operators
    * `LoadImage`: Read image files
    * `LoadVideo`: Read video files
    * `LoadAudio`: Read audio files
    * `LoadGIF`: Read GIF files
    * `LoadTorchPickle`: Read binary files saved by [`torch.save`](https://docs.pytorch.org/docs/stable/generated/torch.save.html) [This operator may cause code injection attacks in binary files, please use with caution!]
* Media file processing operators
    * `ImageCropAndResize`: Crop and resize images
* Meta operators
    * `SequencialProcess`: Route each data in the sequence to an operator
    * `RouteByExtensionName`: Route to specific operators by file extension
    * `RouteByType`: Route to specific operators by data type

### Operator Usage

Data operators are connected with the `>>` symbol to form data processing pipelines, for example:

```python
from diffsynth.core.data.operators import *

data = "image.jpg"
data_pipeline = ToAbsolutePath(base_path="/data") >> LoadImage() >> ImageCropAndResize(max_pixels=512*512)
data = data_pipeline(data)
```

After passing through each operator, the data is processed in sequence:

* `ToAbsolutePath(base_path="/data")`: `"/data/image.jpg"`
* `LoadImage()`: `<PIL.Image.Image image mode=RGB size=1024x1024 at 0x7F8E7AAEFC10>`
* `ImageCropAndResize(max_pixels=512*512)`: `<PIL.Image.Image image mode=RGB size=512x512 at 0x7F8E7A936F20>`

We can compose functionally complete data pipelines, for example, the default video data operator for the universal dataset is:

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

It includes the following logic:

* If the data is of type `str`
    * If it's a `"jpg", "jpeg", "png", "webp"` type file
        * Load this image
        * Crop and scale to a specific resolution
        * Pack into a list, treating it as a single-frame video
    * If it's a `"gif"` type file
        * Load the GIF file content
        * Crop and scale each frame to a specific resolution
    * If it's a `"mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"` type file
        * Load the video file content
        * Crop and scale each frame to a specific resolution
* If the data is not of type `str`, an error is reported

## Universal Dataset

`diffsynth.core.data` provides a unified dataset implementation. The dataset requires the following parameters:

* `base_path`: Root directory. If the dataset contains relative paths to image files, this field needs to be filled in to load the files pointed to by these paths
* `metadata_path`: Metadata directory, records the file paths of all metadata, supports `csv`, `json`, `jsonl` formats
* `repeat`: Data repetition count, defaults to 1, this parameter affects the number of training steps in an epoch
* `data_file_keys`: Data field names that need to be loaded, for example `(image, edit_image)`
* `main_data_operator`: Main loading operator, needs to assemble the data processing pipeline through data processing operators
* `special_operator_map`: Special operator mapping, operator mappings built for fields that require special processing

### Metadata

The dataset's `metadata_path` points to a metadata file, supporting `csv`, `json`, `jsonl` formats. The following provides examples:

* `csv` format: High readability, does not support list data, small memory footprint

```csv
image,prompt
image_1.jpg,"a dog"
image_2.jpg,"a cat"
```

* `json` format: High readability, supports list data, large memory footprint

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

* `jsonl` format: Low readability, supports list data, small memory footprint

```json
{"image": "image_1.jpg", "prompt": "a dog"}
{"image": "image_2.jpg", "prompt": "a cat"}
```

How to choose the best metadata format?

* If the data volume is large, reaching tens of millions, since `json` file parsing requires additional memory, it's not available. Please use `csv` or `jsonl` format
* If the dataset contains list data, such as edit models that require multiple images as input, since `csv` format cannot store list format data, it's not available. Please use `json` or `jsonl` format

### Data Loading Logic

When no additional settings are made, the dataset defaults to outputting data from the metadata set. Image and video file paths will be output in string format. To load these files, you need to set `data_file_keys`, `main_data_operator`, and `special_operator_map`.

In the data processing flow, processing is done according to the following logic:
* If the field is in `special_operator_map`, call the corresponding operator in `special_operator_map` for processing
* If the field is not in `special_operator_map`
    * If the field is in `data_file_keys`, call the `main_data_operator` operator for processing
    * If the field is not in `data_file_keys`, no processing is done

`special_operator_map` can be used to implement special data processing. For example, in the model [Wan-AI/Wan2.2-Animate-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B), the input character face video `animate_face_video` is processed at a fixed resolution, inconsistent with the output video. Therefore, this field is processed by a dedicated operator:

```python
special_operator_map={
    "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
}
```

### Other Notes

When the data volume is too small, you can appropriately increase `repeat` to extend the training time of a single epoch, avoiding frequent model saving that generates considerable overhead.

When data volume * `repeat` exceeds $10^9$, we observe that the dataset speed becomes significantly slower. This seems to be a `PyTorch` bug, and we are not sure if newer versions of `PyTorch` have fixed this issue.