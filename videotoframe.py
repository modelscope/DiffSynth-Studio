import cv2
import os

def video_to_frames(video_path, output_dir, every_n_frames=1):
    # 建立輸出資料夾（如果尚未存在）
    os.makedirs(output_dir, exist_ok=True)

    # 開啟影片檔
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 無法開啟影片:", video_path)
        return

    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % every_n_frames == 0:
            # 儲存圖片幀
            filename = f"{output_dir}/frame_{saved_count:05d}.png"
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ 完成：共擷取 {saved_count} 幀並儲存於 {output_dir}")



# 用法範例
# video_to_frames("dog.mp4", "output_frames_dog", every_n_frames=1)
cap = cv2.VideoCapture("styledata/video/dog.mp4")
if not cap.isOpened():
    print("can't open video")

orig_fps = cap.get(cv2.CAP_PROP_FPS)
print("原影片 fps =", orig_fps)
