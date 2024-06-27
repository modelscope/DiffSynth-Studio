from PIL import ImageEnhance
from .base import VideoProcessor


class ContrastEditor(VideoProcessor):
    def __init__(self, rate=1.5):
        self.rate = rate

    @staticmethod
    def from_model_manager(model_manager, **kwargs):
        return ContrastEditor(**kwargs)
    
    def __call__(self, rendered_frames, **kwargs):
        rendered_frames = [ImageEnhance.Contrast(i).enhance(self.rate) for i in rendered_frames]
        return rendered_frames


class SharpnessEditor(VideoProcessor):
    def __init__(self, rate=1.5):
        self.rate = rate

    @staticmethod
    def from_model_manager(model_manager, **kwargs):
        return SharpnessEditor(**kwargs)
    
    def __call__(self, rendered_frames, **kwargs):
        rendered_frames = [ImageEnhance.Sharpness(i).enhance(self.rate) for i in rendered_frames]
        return rendered_frames
