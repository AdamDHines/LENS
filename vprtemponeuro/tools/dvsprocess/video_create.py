import cv2
import os


def create_video_from_frames(frame_folder, output_file, fps=30):
    frame_files = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.png')]
    
    if not frame_files:
        raise ValueError("No frames found in the specified folder.")

    frame_files.sort(key=lambda x: int(x.split('/')[-1].split('_')[-1].replace('.png', '')))

    # Explicitly read the first frame in color
    first_frame = cv2.imread(frame_files[0], cv2.IMREAD_COLOR)
    if first_frame is None:
        raise ValueError("Failed to read the first frame.")

    height, width, layers = first_frame.shape

    # Using 'XVID' codec for AVI format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        video.write(frame)

    video.release()
    print(f"Video has been saved to {output_file}")

# Your specific frame folder and output file path
image_folder = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/test_accum'
output_video_file = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/output_video_expo.avi'

# Call the function with your specific parameters
create_video_from_frames(image_folder, output_video_file, fps=30)
