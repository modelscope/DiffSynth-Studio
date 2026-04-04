from typing_extensions import Literal, TypeAlias


Processor_id: TypeAlias = Literal[
    "canny", "depth", "softedge", "lineart", "lineart_anime", "openpose", "normal", "tile", "none", "inpaint"
]

class Annotator:
    def __init__(self, processor_id: Processor_id, model_path="models/Annotators", detect_resolution=None, device='cuda', skip_processor=False):
        if not skip_processor:
            if processor_id == "canny":
                from controlnet_aux.processor import CannyDetector
                self.processor = CannyDetector()
            elif processor_id == "depth":
                from controlnet_aux.processor import MidasDetector
                self.processor = MidasDetector.from_pretrained(model_path).to(device)
            elif processor_id == "softedge":
                from controlnet_aux.processor import HEDdetector
                self.processor = HEDdetector.from_pretrained(model_path).to(device)
            elif processor_id == "lineart":
                from controlnet_aux.processor import LineartDetector
                self.processor = LineartDetector.from_pretrained(model_path).to(device)
            elif processor_id == "lineart_anime":
                from controlnet_aux.processor import LineartAnimeDetector
                self.processor = LineartAnimeDetector.from_pretrained(model_path).to(device)
            elif processor_id == "openpose":
                from controlnet_aux.processor import OpenposeDetector
                self.processor = OpenposeDetector.from_pretrained(model_path).to(device)
            elif processor_id == "normal":
                from controlnet_aux.processor import NormalBaeDetector
                self.processor = NormalBaeDetector.from_pretrained(model_path).to(device)
            elif processor_id == "tile" or processor_id == "none" or processor_id == "inpaint":
                self.processor = None
            else:
                raise ValueError(f"Unsupported processor_id: {processor_id}")
        else:
            self.processor = None

        self.processor_id = processor_id
        self.detect_resolution = detect_resolution
    
    def to(self,device):
        if hasattr(self.processor,"model") and hasattr(self.processor.model,"to"):

            self.processor.model.to(device)

    def __call__(self, image, mask=None):
        width, height = image.size
        if self.processor_id == "openpose":
            kwargs = {
                "include_body": True,
                "include_hand": True,
                "include_face": True
            }
        else:
            kwargs = {}
        if self.processor is not None:
            detect_resolution = self.detect_resolution if self.detect_resolution is not None else min(width, height)
            image = self.processor(image, detect_resolution=detect_resolution, image_resolution=min(width, height), **kwargs)
        image = image.resize((width, height))
        return image

