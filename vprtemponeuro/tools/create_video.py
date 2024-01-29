# Imports
import os
import cv2

from tqdm import tqdm

class CreateVideo():
    def __init__(self, args):
        super(CreateVideo, self).__init__()
        self.args = args

    def create_video_from_frames(self):
        frame_files = [os.path.join(self.args.dataset_folder,self.args.input_file, f) for f in os.listdir(os.path.join(self.args.dataset_folder,self.args.input_file)) if f.endswith('.png')]
        
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
        output_file = os.path.join(self.args.dataset_folder, self.args.input_file + ".avi")
        video = cv2.VideoWriter(output_file, fourcc, self.args.timebin, (width, height))

        for frame_file in tqdm(frame_files,desc="Creating video from frames"):
            frame = cv2.imread(frame_file, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            video.write(frame)

        video.release()
        print(f"Video has been saved to {output_file}")