#Model Download
from modelscope import snapshot_download
model_dir = snapshot_download('Wan-AI/Wan2.2-S2V-14B',local_dir='./models/Wan-AI/Wan2.2-S2V-14B')