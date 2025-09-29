import cv2
import glob
import os

def frames_to_video(frames_dir, out_path, fps=24):
    files = sorted(glob.glob(os.path.join(frames_dir, "frame*.png")))
    if not files:
        raise RuntimeError("找不到影格！")

    # 讀首張影格以取得尺寸
    first_frame = cv2.imread(files[0])
    h, w, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MPEG‑4
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for f in files:
        frame = cv2.imread(f)
        writer.write(frame)

    writer.release()
    print(f"✅ 影片已輸出：{out_path}")

# 用法
frames_to_video("deadiff_dog", "dogdeadiff.mp4", fps=28)
