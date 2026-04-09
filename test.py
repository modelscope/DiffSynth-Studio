from diffsynth.models.model_loader import ModelPool

pool = ModelPool()
pool.auto_load_model("models/jd-opensource/JoyAI-Image-Edit/vae/Wan2.1_VAE.pth")
model = pool.fetch_model("wan_video_vae")