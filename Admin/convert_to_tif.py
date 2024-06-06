import os
import glob
import cv2

image_folder = r"C:\Users\chloe\DE4\Masters\Dataset\All_Data"

jpg_images = glob.glob(r"C:\Users\chloe\DE4\Masters\Dataset\All_Data\**\*_8.jpg")

for i, filename in enumerate(jpg_images):
        # Get filename without extension
        remove_extension = filename.split('.')[0]
        # Rename with .tif extension
        os.rename(filename, remove_extension + ".tif")

jpg_images = glob.glob(r"C:\Users\chloe\DE4\Masters\Dataset\allImagesMasks\*_s.jpg")
print(jpg_images)