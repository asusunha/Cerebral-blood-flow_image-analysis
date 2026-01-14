import cv2
import os

video_path = r"D:\VIDEO\M-24\Video_00056_00052.mp4"
save_dir = r"D:\VIDEO\M-24\Video_00056_00052_2m30sto2m35s_tiff"

os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
start_sec = 150
end_sec = 155

start_frame = int(start_sec * fps)
end_frame = int(end_sec * fps)

print(f"FPS: {fps:.2f}")
print(f"Extract frames: {start_frame} ~ {end_frame}")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_idx = start_frame
save_idx = 0

while cap.isOpened() and frame_idx < end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    filename = f"frame_{save_idx:04d}.tiff"
    cv2.imwrite(os.path.join(save_dir, filename), frame)

    frame_idx += 1
    save_idx += 1

cap.release()
print(f"완료: {save_idx} frames TIFF 저장됨")
