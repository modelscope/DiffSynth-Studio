"""Gaussian-blur track for a whole video: one mp4 in, one blurred mp4 out.

The counterpart of video_lineart: that one draws outlines, this one degrades the
frames so a restoration ControlNet has something to restore. Resolution, frame
count and fps are unchanged, so input and output stay pixel-aligned.

ffmpeg rather than imageio for two reasons: imageio's writer drops the audio
track, which the training loader reads from this very file, and the batch
blurring scripts use the same ffmpeg command, so single-file and batch results
match bit for bit.

gblur is a recursive IIR approximation (Alvarez-Mazorra), not an FIR
convolution, so `steps` is the number of passes rather than a kernel size.
Measured sigma is accurate (5.0 nominal -> 5.026) and the shape is within ~1%
of a true Gaussian.
"""
import argparse
import os
import subprocess

DEFAULT_SIGMA = 5.0
STEPS = 3     # within 0.7% of a true Gaussian already; more passes buy little
CRF = 16


def ffmpeg_bin():
    """System ffmpeg if present, else the static build inside imageio_ffmpeg."""
    from shutil import which
    path = which("ffmpeg")
    if path is None:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
    return path


def video_to_blur(input_path, output_path, sigma=DEFAULT_SIGMA, steps=STEPS, crf=CRF):
    """Blur every frame with sigma, copying the audio track through untouched."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = [
        ffmpeg_bin(), "-nostdin", "-v", "error", "-y",
        "-i", input_path,
        "-vf", f"gblur=sigma={sigma}:steps={steps}",
        "-map", "0:v:0", "-map", "0:a:0?",          # the ? tolerates a file with no audio
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr.strip()}")
    return output_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("sigma", type=float, nargs="?", default=DEFAULT_SIGMA)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--crf", type=int, default=CRF)
    args = p.parse_args()
    path = video_to_blur(args.input, args.output, sigma=args.sigma,
                         steps=args.steps, crf=args.crf)
    print(f"blur sigma={args.sigma:g} -> {path}", flush=True)


if __name__ == "__main__":
    main()
