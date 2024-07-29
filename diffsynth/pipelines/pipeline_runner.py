import os, torch, json
from .sd_video import ModelManager, SDVideoPipeline, ControlNetConfigUnit
from ..processors.sequencial_processor import SequencialProcessor
from ..data import VideoData, save_frames, save_video



class SDVideoPipelineRunner:
    def __init__(self, in_streamlit=False):
        self.in_streamlit = in_streamlit


    def load_pipeline(self, model_list, textual_inversion_folder, device, lora_alphas, controlnet_units):
        # Load models
        model_manager = ModelManager(torch_dtype=torch.float16, device=device)
        model_manager.load_models(model_list)
        pipe = SDVideoPipeline.from_model_manager(
            model_manager,
            [
                ControlNetConfigUnit(
                    processor_id=unit["processor_id"],
                    model_path=unit["model_path"],
                    scale=unit["scale"]
                ) for unit in controlnet_units
            ]
        )
        textual_inversion_paths = []
        for file_name in os.listdir(textual_inversion_folder):
            if file_name.endswith(".pt") or file_name.endswith(".bin") or file_name.endswith(".pth") or file_name.endswith(".safetensors"):
                textual_inversion_paths.append(os.path.join(textual_inversion_folder, file_name))
        pipe.prompter.load_textual_inversions(textual_inversion_paths)
        return model_manager, pipe
    

    def load_smoother(self, model_manager, smoother_configs):
        smoother = SequencialProcessor.from_model_manager(model_manager, smoother_configs)
        return smoother


    def synthesize_video(self, model_manager, pipe, seed, smoother, **pipeline_inputs):
        torch.manual_seed(seed)
        if self.in_streamlit:
            import streamlit as st
            progress_bar_st = st.progress(0.0)
            output_video = pipe(**pipeline_inputs, smoother=smoother, progress_bar_st=progress_bar_st)
            progress_bar_st.progress(1.0)
        else:
            output_video = pipe(**pipeline_inputs, smoother=smoother)
        model_manager.to("cpu")
        return output_video


    def load_video(self, video_file, image_folder, height, width, start_frame_id, end_frame_id):
        video = VideoData(video_file=video_file, image_folder=image_folder, height=height, width=width)
        if start_frame_id is None:
            start_frame_id = 0
        if end_frame_id is None:
            end_frame_id = len(video)
        frames = [video[i] for i in range(start_frame_id, end_frame_id)]
        return frames


    def add_data_to_pipeline_inputs(self, data, pipeline_inputs):
        pipeline_inputs["input_frames"] = self.load_video(**data["input_frames"])
        pipeline_inputs["num_frames"] = len(pipeline_inputs["input_frames"])
        pipeline_inputs["width"], pipeline_inputs["height"] = pipeline_inputs["input_frames"][0].size
        if len(data["controlnet_frames"]) > 0:
            pipeline_inputs["controlnet_frames"] = [self.load_video(**unit) for unit in data["controlnet_frames"]]
        return pipeline_inputs


    def save_output(self, video, output_folder, fps, config):
        os.makedirs(output_folder, exist_ok=True)
        save_frames(video, os.path.join(output_folder, "frames"))
        save_video(video, os.path.join(output_folder, "video.mp4"), fps=fps)
        config["pipeline"]["pipeline_inputs"]["input_frames"] = []
        config["pipeline"]["pipeline_inputs"]["controlnet_frames"] = []
        with open(os.path.join(output_folder, "config.json"), 'w') as file:
            json.dump(config, file, indent=4)


    def run(self, config):
        if self.in_streamlit:
            import streamlit as st
        if self.in_streamlit: st.markdown("Loading videos ...")
        config["pipeline"]["pipeline_inputs"] = self.add_data_to_pipeline_inputs(config["data"], config["pipeline"]["pipeline_inputs"])
        if self.in_streamlit: st.markdown("Loading videos ... done!")
        if self.in_streamlit: st.markdown("Loading models ...")
        model_manager, pipe = self.load_pipeline(**config["models"])
        if self.in_streamlit: st.markdown("Loading models ... done!")
        if "smoother_configs" in config:
            if self.in_streamlit: st.markdown("Loading smoother ...")
            smoother = self.load_smoother(model_manager, config["smoother_configs"])
            if self.in_streamlit: st.markdown("Loading smoother ... done!")
        else:
            smoother = None
        if self.in_streamlit: st.markdown("Synthesizing videos ...")
        output_video = self.synthesize_video(model_manager, pipe, config["pipeline"]["seed"], smoother, **config["pipeline"]["pipeline_inputs"])
        if self.in_streamlit: st.markdown("Synthesizing videos ... done!")
        if self.in_streamlit: st.markdown("Saving videos ...")
        self.save_output(output_video, config["data"]["output_folder"], config["data"]["fps"], config)
        if self.in_streamlit: st.markdown("Saving videos ... done!")
        if self.in_streamlit: st.markdown("Finished!")
        video_file = open(os.path.join(os.path.join(config["data"]["output_folder"], "video.mp4")), 'rb')
        if self.in_streamlit: st.video(video_file.read())
