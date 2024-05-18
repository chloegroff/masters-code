import numpy as np
from scipy.ndimage import rotate, zoom
from random import shuffle, choice
from os import remove
from os.path import sep
from glob import glob as glob
import numpy.typing as npt
import cv2
from numpy.random import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pkl

def generate_training(
        image_path: str, save_path: str, num_images: int, resolution: tuple[int, int], prop_normal: float,
        prop_rot: float, prop_crop: float, prop_zoom: float, 
        zoom_range: tuple[int, int] = (0.5, 3), crop_range: tuple[int, int] = [0.1, 0.6]
        ):
    
    # Clears files in save directory
    prev_files = glob(save_path + sep + '*')
    for f in prev_files:
        remove(f)
    
    # Gets mask and image files
    img_files = glob(image_path + sep + '*.jpg')
    mask_files = glob(image_path + sep + '*_s.*')

    # Takes last few charactaers of each mask path
    mask_stamps = [i[:-6] for i in mask_files]

    # checks if each image has a mask, appends images that do to a list
    masked_imgs = []
    for i in img_files:
        if i[:-4] in mask_stamps:
            masked_imgs.append(i)
    
    # Zips images and masks then randomly shuffles list
    zip_files = list(zip(masked_imgs, mask_files))
    shuffle(zip_files)

    # Loops through all masked images unitl either length of list is reached or desired prop of total normal images saved to dir
    index = 0
    #pbar = tqdm(total=num_images)
    
    
    for i in range(int(num_images*prop_normal)):
        if i >= len(zip_files) - 1:
            break

        # Loads image and mask
        img = cv2.imread(zip_files[i][0])[:,500:-600]
        mask = cv2.imread(zip_files[i][1], 0)[:,500:-600]

        # Scales to desired size
        scales = [resolution[0]/(img.shape[0]),resolution[1]/(img.shape[1]), 1]
        img = zoom(img, scales)
        mask = zoom(mask, [scales[0], scales[1]])

        if np.sum(mask)/255/mask.size >= 0.0 and np.sum(mask)/255/mask.size <= 0.01:
            mask = np.where(mask > 128, 255,0)
            mask = np.array(mask, dtype=np.uint8)
            # saves mask and image
            cv2.imwrite(save_path + sep +  f'{index}_' + zip_files[i][0].split(sep)[-1][:-4] + '_i.tif', img)
            cv2.imwrite(save_path + sep + f'{index}_' + zip_files[i][1].split(sep)[-1], mask)
            index += 1
            #pbar.update(1)

        continue
    print('num training images with less than 1% white: ', index)
    # Loops until desired number of images reached
    while index < num_images:
        flag = False

        # Check if image has been manipulated
        while not flag:
            
            # selects random image and mask and loads
            im_file, mask_file = choice(zip_files)      
            img = cv2.imread(im_file)[:,550:-600]
            mask = cv2.imread(mask_file, 0)[:,550:-600]

            if np.sum(mask)/255/mask.size >= 0.0 and np.sum(mask)/255/mask.size <= 0.01:
                # Rescales to resolution
                scales = [resolution[0]/(img.shape[0]),resolution[1]/(img.shape[1]), 1]
                img = zoom(img, scales)
                mask = zoom(mask, [scales[0], scales[1]])

                # if prop_zoom > random():
                #     scale = random()*np.diff(zoom_range) + min(zoom_range)

                #     mask = zoom(mask, scale[0])   
                #     img = zoom(img, scale[0])

                #     img = img

                #     print('zoom')
                #     plt.imshow(img)
                #     plt.show()

                # Crops image
                if prop_crop > random():
                    cut = (random()*np.diff(crop_range) + min(crop_range))
                    
                    img_slice = img[:int(cut*img.shape[0])]
                    
                    fill_color = np.mean(img_slice, keepdims=2)
                    
                    img[:int(cut*img.shape[0])] = fill_color
                    mask[:int(cut*img.shape[0])] = 0
                    flag = True



                
                # rotates image
                if prop_rot > random():
                    angle = random()*360
                    img = rotate(img, angle, reshape= False , mode= 'nearest')
                    mask = rotate(mask, angle, reshape= False , prefilter= True)
                    flag = True

                # if np.sum(mask)/255/mask.size < 0.0:
                #     flag = False


        # Sets mask to binary
        mask = np.where(mask > 128, 255,0)
        mask = np.array(mask, dtype=np.uint8)

        # Saves mask and image
        cv2.imwrite(save_path + sep + f'{index}_' + im_file.split(sep)[-1][:-4] + '_i.tif', img)
        cv2.imwrite(save_path + sep + f'{index}_' + mask_file.split(sep)[-1], mask)
        
        #pbar.update(1)
        index += 1
    #pbar.close()

if __name__ == '__main__':
    generate_training(image_path = r'C:\Users\chloe\DE4\Masters\Dataset\allImagesMasks', 
                      save_path = r'C:\Users\chloe\DE4\Masters\Dataset\Training_Data_small_nothing', 
                      num_images = 250, 
                      resolution = (224,224), 
                      prop_normal = 1, prop_rot = 0.5, prop_crop = 0.5, prop_zoom = 0)

