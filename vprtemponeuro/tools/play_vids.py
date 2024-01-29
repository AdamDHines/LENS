import cv2
import threading
import time

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1.0 / fps

    cv2.namedWindow(video_path, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(video_path, 640, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(video_path, frame)

        # Wait for either a frame duration or a key press, whichever is shorter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(max(frame_duration - (1.0 / 1000.0), 0))

    cap.release()
    cv2.destroyAllWindows()

# Paths to your video files
video1_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/20200422_172431-sunset2_concat.mp4'
video2_path = '/home/adam/repo/rpg_e2vid/scripts/extracted_data/dvs_reference_sunset2.mp4'

# Create threads for each video
thread1 = threading.Thread(target=play_video, args=(video1_path,))
thread2 = threading.Thread(target=play_video, args=(video2_path,))

# Start threads
thread1.start()
thread2.start()

