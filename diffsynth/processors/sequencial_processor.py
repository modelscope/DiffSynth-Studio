from .base import VideoProcessor


class AutoVideoProcessor(VideoProcessor):
    def __init__(self):
        pass

    @staticmethod
    def from_model_manager(model_manager, processor_type, **kwargs):
        if processor_type == "FastBlend":
            from .FastBlend import FastBlendSmoother
            return FastBlendSmoother.from_model_manager(model_manager, **kwargs)
        elif processor_type == "Contrast":
            from .PILEditor import ContrastEditor
            return ContrastEditor.from_model_manager(model_manager, **kwargs)
        elif processor_type == "Sharpness":
            from .PILEditor import SharpnessEditor
            return SharpnessEditor.from_model_manager(model_manager, **kwargs)
        elif processor_type == "RIFE":
            from .RIFE import RIFESmoother
            return RIFESmoother.from_model_manager(model_manager, **kwargs)
        else:
            raise ValueError(f"invalid processor_type: {processor_type}")


class SequencialProcessor(VideoProcessor):
    def __init__(self, processors=[]):
        self.processors = processors

    @staticmethod
    def from_model_manager(model_manager, configs):
        processors = [
            AutoVideoProcessor.from_model_manager(model_manager, config["processor_type"], **config["config"])
            for config in configs
        ]
        return SequencialProcessor(processors)
    
    def __call__(self, rendered_frames, **kwargs):
        for processor in self.processors:
            rendered_frames = processor(rendered_frames, **kwargs)
        return rendered_frames
