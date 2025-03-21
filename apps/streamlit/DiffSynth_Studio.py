# Set web page format
import streamlit as st
st.set_page_config(layout="wide")
# Configure GPU memory usage based on available hardware
import torch
import platform

# Check for CUDA (NVIDIA GPUs)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.999, 0)
    device = "cuda"
# Check for MPS (Apple Silicon)
elif hasattr(torch, 'mps') and torch.backends.mps.is_available() and platform.processor() == 'arm':
    device = "mps"
else:
    device = "cpu"

st.markdown(f"""
# DiffSynth Studio

[Source Code](https://github.com/modelscope/DiffSynth-Studio)

Welcome to DiffSynth Studio. Running on: {device.upper()}
""")
