''' Use this to move the images and associated masks from a specific box into a different folder.
Can be used to split a train and validation data set or to make timelapse videos. '''

import os
import glob
import cv2
from os.path import sep

box_num = 8

image_folder = rf"C:\Users\chloe\DE4\Masters\Dataset\allImagesMasks_del{box_num}"
save_folder = rf"C:\Users\chloe\DE4\Masters\Dataset\validation_del{box_num}"

box_img = glob.glob(rf"C:\Users\chloe\DE4\Masters\Dataset\allImagesMasks_del{box_num}\*_{box_num}.jpg")
#box8_masks = glob.glob(r"C:\Users\chloe\DE4\Masters\Dataset\allImagesMasks_del8\*_8_s.tif")
print(len(box_img))

# Clears files in save directory
prev_files = glob.glob(save_folder + sep + '*')
for f in prev_files:
        os.remove(f)

for i, filename in enumerate(box_img):
        # Get path of file and associated mask
        img_path_rmv_jpg = filename.split('.')[0]
        mask_path = img_path_rmv_jpg + '_s.tif'
        # Get image and mask name
        image_name = os.path.relpath(filename, image_folder)
        mask_name = os.path.relpath(mask_path, image_folder)
        # Move box8 images to new folder
        img = cv2.imread(filename)
        mask = cv2.imread(mask_path)
        # print(filename)
        cv2.imwrite(os.path.join(save_folder, image_name), img)
        cv2.imwrite(os.path.join(save_folder, mask_name), mask)
        os.remove(filename)
        os.remove(mask_path)

box_img = glob.glob(rf"C:\Users\chloe\DE4\Masters\Dataset\allImagesMasks_del{box_num}\*_{box_num}.jpg")
print(len(box_img))