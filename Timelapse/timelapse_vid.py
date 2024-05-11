import os
import imageio
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def add_subtitle(image_path, subtitle_text):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", size=120)  # Change the font size here

    # Get the size of the image
    img_width, img_height = img.size

    # Get the size of the text
    text_width, text_height = draw.textsize(subtitle_text, font)

    # Calculate the position to place the text slightly above the bottom of the image
    text_position = ((img_width - text_width) // 2, img_height - text_height - 50)

    # Draw the text in yellow color
    draw.text(text_position, subtitle_text, fill="black", font=font)

    return img

def create_timelapse(folder_path, output_file, frame_duration=2):
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    images.sort()

    frame_rate = 1 / frame_duration

    # Determine the size of the video frame based on the first image's dimensions
    first_image = Image.open(os.path.join(folder_path, images[0]))
    frame_width, frame_height = first_image.size

    video_writer = imageio.get_writer(output_file, fps=frame_rate)

    # Get the height of the subtitle text
    _, subtitle_height = ImageDraw.Draw(Image.new('RGB', (1, 1))).textsize("Subtitle", font=ImageFont.truetype("arial.ttf", size=80))

    for image in images:
        image_path = os.path.join(folder_path, image)

        # Show only the first 8 characters of the filename as the subtitle
        subtitle_text = os.path.splitext(image)[0][:8]

        # Add the subtitle to the image
        subtitle_image = add_subtitle(image_path, subtitle_text)

        # Resize the image to fit the video frame
        resized_image = subtitle_image.resize((frame_width, frame_height))

        # Convert the PIL image to NumPy array and write to the video
        frame_array = np.array(resized_image)
        video_writer.append_data(frame_array)

    video_writer.close()

if __name__ == "__main__":
    input_folder = r"D:\William\Design_Engineering\Year 3\Research Placement\Post-Image Analysis\Lidinoid magenta 1"
    output_video = "output_video.mp4"  # Changed the file extension to MP4

    create_timelapse(input_folder, output_video)
