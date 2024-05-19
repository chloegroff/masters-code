import os
import glob
import cv2
import numpy as np

index = 0
image_name = ''

image_folder = r"C:\Users\chloe\DE4\Masters\Dataset\allImagesMasks"
masks = glob.glob(r"C:\Users\chloe\DE4\Masters\Dataset\allImagesMasks\*_s.tif")
save_folder = r'C:\Users\chloe\DE4\Masters\Dataset\allImagesMasks_1plus'
#box8_masks = glob.glob(r"C:\Users\chloe\DE4\Masters\Dataset\allImagesMasks_del8\*_8_s.tif")
print(len(masks))

for i, mask_path in enumerate(masks):
        # Find associated image with mask
        mask_name = os.path.relpath(mask_path, image_folder)
        image_array = np.array(mask_name.split('_')[:6])
        image_name = '_'.join(image_array) + '.jpg'
        #image_name = np.core.defchararray.join('_', image_array) + '.jpg'
        # Read the image and mask pair
        mask = cv2.imread(mask_path)
        img = cv2.imread(os.path.join(image_folder, image_name))
        # Write masks (and associated image) that are more than 1% white pixels (small+ growth)
        if np.sum(mask)/255/mask.size > 0.01:
            index += 1
            cv2.imwrite(os.path.join(save_folder, image_name), img)
            cv2.imwrite(os.path.join(save_folder, mask_name), mask)

print(index)

