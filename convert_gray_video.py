import cv2

# 輸入與輸出影片檔名
input_video_path = 'dog.mp4'   # 請將此路徑換成你的原始影片
output_video_path = 'output_gray_video.mp4'

# 開啟影片
cap = cv2.VideoCapture(input_video_path)

# 確認影片是否成功開啟
if not cap.isOpened():
    print("無法開啟影片")
    exit()

# 取得影片的寬度、高度與 FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 建立 VideoWriter 物件，編碼格式使用 mp4v
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 轉成灰階
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 寫入灰階影格
    out.write(gray_frame)

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()

print("轉換完成，儲存為:", output_video_path)