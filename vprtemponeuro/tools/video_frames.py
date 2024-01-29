import cv2
import os
import matplotlib.pyplot as plt

def extract_frames_per_second_with_crop(video_path, output_folder, crop_bottom_pixels):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the original FPS and frame size of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    image_count = 0
    displayed_sample = False

    while True:
        # Read the next frame
        ret, frame = cap.read()

        # Break the loop if there are no frames left
        if not ret:
            break

        # Crop the bottom of the frame
        cropped_frame = frame[:-crop_bottom_pixels, :]

        # Display a sample frame
        if not displayed_sample:
            plt.imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            plt.title(f'Sample Cropped Frame (Image {image_count})')
            plt.show()
            displayed_sample = True  # Set to False to see more samples

        # Extract one frame per second
        #if frame_count % int(original_fps) == 0:
            # Save the frame as an image file
        filename = f"{output_folder}/images_{image_count:04d}.png"
        cv2.imwrite(filename, cropped_frame)
        image_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extraction completed. {image_count} images saved in '{output_folder}'.")

# Example usage
video_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/20200421_170039-sunset1_concat.mp4'
output_folder = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/sunset1'
crop_bottom_pixels = 125  # Adjust the number of pixels to crop from the bottom
extract_frames_per_second_with_crop(video_path, output_folder, crop_bottom_pixels)
