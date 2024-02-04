from .base import VideoProcessor


class SequencialProcessor(VideoProcessor):
    def __init__(self, processors=[]):
        self.processors = processors

    @staticmethod
    def from_model_manager(model_manager, **kwargs):
        return SequencialProcessor(**kwargs)
    
    def __call__(self, rendered_frames, **kwargs):
        for processor in self.processors:
            rendered_frames = processor(rendered_frames, **kwargs)
        return rendered_frames