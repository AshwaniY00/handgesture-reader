import cv2
import os

video_dir = "videos/"
output_base = "ISL_Dataset/"
frame_size = (64, 64)

for filename in os.listdir(video_dir):
    if filename.endswith(".mp4"):
        label = os.path.splitext(filename)[0].split("_")[0]
  # Extract 'A' from 'A_gesture.mp4'
        output_folder = os.path.join(output_base, label)
        os.makedirs(output_folder, exist_ok=True)

        video_path = os.path.join(video_dir, filename)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, frame_size)
            frame_path = os.path.join(output_folder, f"{label}_frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        cap.release()
        print(f"✅ {filename}: Saved {frame_count} frames to {output_folder}")
