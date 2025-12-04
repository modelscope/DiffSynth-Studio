import imageio, os
import numpy as np
from PIL import Image
from tqdm import tqdm
import subprocess
import shutil


class LowMemoryVideo:
    def __init__(self, file_name):
        self.reader = imageio.get_reader(file_name)
    
    def __len__(self):
        return self.reader.count_frames()

    def __getitem__(self, item):
        return Image.fromarray(np.array(self.reader.get_data(item))).convert("RGB")

    def __del__(self):
        self.reader.close()


def split_file_name(file_name):
    result = []
    number = -1
    for i in file_name:
        if ord(i)>=ord("0") and ord(i)<=ord("9"):
            if number == -1:
                number = 0
            number = number*10 + ord(i) - ord("0")
        else:
            if number != -1:
                result.append(number)
                number = -1
            result.append(i)
    if number != -1:
        result.append(number)
    result = tuple(result)
    return result


def search_for_images(folder):
    file_list = [i for i in os.listdir(folder) if i.endswith(".jpg") or i.endswith(".png")]
    file_list = [(split_file_name(file_name), file_name) for file_name in file_list]
    file_list = [i[1] for i in sorted(file_list)]
    file_list = [os.path.join(folder, i) for i in file_list]
    return file_list


class LowMemoryImageFolder:
    def __init__(self, folder, file_list=None):
        if file_list is None:
            self.file_list = search_for_images(folder)
        else:
            self.file_list = [os.path.join(folder, file_name) for file_name in file_list]
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        return Image.open(self.file_list[item]).convert("RGB")

    def __del__(self):
        pass


def crop_and_resize(image, height, width):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = (image_width - croped_width) // 2
        image = image[:, left: left+croped_width]
        image = Image.fromarray(image).resize((width, height))
    else:
        croped_height = int(image_width / width * height)
        left = (image_height - croped_height) // 2
        image = image[left: left+croped_height, :]
        image = Image.fromarray(image).resize((width, height))
    return image


class VideoData:
    def __init__(self, video_file=None, image_folder=None, height=None, width=None, **kwargs):
        if video_file is not None:
            self.data_type = "video"
            self.data = LowMemoryVideo(video_file, **kwargs)
        elif image_folder is not None:
            self.data_type = "images"
            self.data = LowMemoryImageFolder(image_folder, **kwargs)
        else:
            raise ValueError("Cannot open video or image folder")
        self.length = None
        self.set_shape(height, width)

    def raw_data(self):
        frames = []
        for i in range(self.__len__()):
            frames.append(self.__getitem__(i))
        return frames

    def set_length(self, length):
        self.length = length

    def set_shape(self, height, width):
        self.height = height
        self.width = width

    def __len__(self):
        if self.length is None:
            return len(self.data)
        else:
            return self.length

    def shape(self):
        if self.height is not None and self.width is not None:
            return self.height, self.width
        else:
            height, width, _ = self.__getitem__(0).shape
            return height, width

    def __getitem__(self, item):
        frame = self.data.__getitem__(item)
        width, height = frame.size
        if self.height is not None and self.width is not None:
            if self.height != height or self.width != width:
                frame = crop_and_resize(frame, self.height, self.width)
        return frame

    def __del__(self):
        pass

    def save_images(self, folder):
        os.makedirs(folder, exist_ok=True)
        for i in tqdm(range(self.__len__()), desc="Saving images"):
            frame = self.__getitem__(i)
            frame.save(os.path.join(folder, f"{i}.png"))


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()

def save_frames(frames, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, frame in enumerate(tqdm(frames, desc="Saving images")):
        frame.save(os.path.join(save_path, f"{i}.png"))


def merge_video_audio(video_path: str, audio_path: str):
    # TODO: may need a in-python implementation to avoid subprocess dependency
    """
    Merge the video and audio into a new video, with the duration set to the shorter of the two,
    and overwrite the original video file.

    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the audio file
    """

    # check
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        # create ffmpeg command
        command = [
            'ffmpeg',
            '-y',  # overwrite
            '-i',
            video_path,
            '-i',
            audio_path,
            '-c:v',
            'copy',  # copy video stream
            '-c:a',
            'aac',  # use AAC audio encoder
            '-b:a',
            '192k',  # set audio bitrate (optional)
            '-map',
            '0:v:0',  # select the first video stream
            '-map',
            '1:a:0',  # select the first audio stream
            '-shortest',  # choose the shortest duration
            temp_output
        ]

        # execute the command
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # check result
        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            print(error_msg)
            raise RuntimeError(error_msg)

        shutil.move(temp_output, video_path)
        print(f"Merge completed, saved to {video_path}")

    except Exception as e:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"merge_video_audio failed with error: {e}")


def save_video_with_audio(frames, save_path, audio_path, fps=16, quality=9, ffmpeg_params=None):
    save_video(frames, save_path, fps, quality, ffmpeg_params)
    merge_video_audio(save_path, audio_path)
