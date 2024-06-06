from tensorflow import keras
import tensorflow as tf
import matplotlib
import matplotlib.image as Image
from os.path import sep
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import zoom
from scipy.optimize import curve_fit
import numpy as np
import cv2
import numpy.typing as npt
import json
import os
import string


# def dice_coef(img, img2):
#         if img.shape != img2.shape:
#             raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
#         else:
            
#             lenIntersection=0
            
#             for i in range(img.shape[0]):
#                 for j in range(img.shape[1]):
#                     if ( np.array_equal(img[i][j],img2[i][j]) ):
#                         lenIntersection+=1
             
#             lenimg=img.shape[0]*img.shape[1]
#             lenimg2=img2.shape[0]*img2.shape[1]  
#             value = (2. * lenIntersection  / (lenimg + lenimg2))
#         return value


def dice_coef(y_true, y_pred, smooth=1):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return np.mean(2. * intersection + smooth) / (union + smooth)


def DICE_COE(mask1, mask2):
    intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3) # for easy reading
    return dice

def get_imgs(img_files: str, hls: bool = True, mask: bool = False, resolution: list[int,int] = [224,224]) -> npt.NDArray:
    imgs = []
    for i, img_file in enumerate(img_files):
        if mask:
            img = cv2.imread(img_file, 0)    

        else:
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if hls and not mask:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        scales = [resolution[0]/(img.shape[0]),resolution[1]/(img.shape[1]), 1]
        if mask:
            img = zoom(img, [scales[0], scales[1]])
        
        else:
            img = zoom(img, scales)

        imgs.append(img)

    return np.array(imgs, dtype= np.uint32)


matplotlib.rcParams.update({'font.size': 12, "font.family": "Times New Roman", 
                            'figure.figsize': [3.6, 3.6], 'figure.dpi': 100, 'savefig.dpi': 300})


alpha = ['a', 'b', 'c', 'd', 'e', 'f']
# dim = 512
val_image = r'C:\Users\chloe\DE4\Masters\Dataset\validation_del8'
image_files = glob.glob(val_image + sep +  '*_8.jpg')
mask_files = glob.glob(val_image + sep + '*_s.tif')

# image_files = [r"C:\Users\chloe\DE4\Masters\Dataset\RF_validation\8_03_08_07_31_58_8_i.tif"]
# mask_files = [r"C:\Users\chloe\DE4\Masters\Dataset\RF_validation\8_03_08_07_31_58_8_s.tif"]

# image_files = []
# mask_files = []
# for i, image_path in enumerate(image_files_all[::4]):
#     image_files.append(image_path)
#     mask_files.append(mask_files_all[i])
# print(len(image_files))
# Define model numbers to compare
models = [(21, 224), (23, 224)]
# Create a string of model numbers seperated by '_' e.g. '8_9_10'
# models_str = ''
# for i in range(len(models)-1):
#     models_str = models_str + str(models[i]) + '_'
# models.str = models_str + str(models[-1])
# print(models_str)

number = 1
long_img = True

save_dict = {}
predictions = []


# Loops through each model specified to generate a prediction for an image
for i, (model, dim) in enumerate(models):
    images = get_imgs(image_files, hls = False, resolution=[dim,dim])
    masks = get_imgs(mask_files, mask = True, resolution=[dim,dim])
    model = rf'C:\Users\chloe\DE4\Masters\Models\Model_{model}.keras'
    loaded_model = keras.models.load_model(model, compile=False)

    result = loaded_model.predict(images / 255)
    result = result > 0.5
    predictions.append(result)

# predictions = np.array(predictions)

index = 0
# Loops through each validation image in folder
dc_array = []
plant_pixels = []
for k, mask in enumerate(masks):
    mask = mask/255
    plant_pixels.append(np.sum(mask))
    dc_list = []
    if long_img == True:
        fig, axes = plt.subplots(ncols = len(models) + 1, nrows = 1)
        fig.set_figwidth(7.2)
        fig.set_figheight(3.6)
    else:
        fig, axes = plt.subplots(ncols = 2, nrows = 3)
        fig.set_figwidth(3.6)
        fig.set_figheight(3.6)


    axes = axes.flatten()
    image_name = os.path.relpath(image_files[k], val_image)

    axes[0].imshow(np.reshape(mask*255, (dim, dim)), cmap="gray")
    axes[0].set_xlabel('Ground Truth')
    for j, mod in enumerate(models):

        axes[j + 1].imshow(np.reshape(predictions[j][k]*255, (mod[1], mod[1])), cmap="gray")
        axes[j + 1].set_xlabel('(' + alpha[j] + ')')
        # ground_truth = np.array(np.reshape(mask[index], (224,224,1)))
        # dc_list.append(dice_coef(np.array(np.reshape(mask, (224,224,1))), np.array(np.reshape(result, (224,224,1))))) 
        if mod[1] != model[1]:
            mask = get_imgs([mask_files[k]], mask=True, resolution=[mod[1], mod[1]])[0]

        dc_list.append(dice_coef(np.array(np.reshape(mask/255, (models[j][1],models[j][1]) )), np.array(np.reshape(predictions[j][k], (models[j][1],models[j][1]) ) )) )


    dc_array.append(dc_list)


    for a, ax in enumerate(axes):
        axes[a].tick_params(
            axis='both',          # changes apply to the voth axis
            which='both', 
            left = False,   # both major and minor ticks are affected
            right = False,
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        
        axes[a].xaxis.set_ticklabels([])
        axes[a].yaxis.set_ticklabels([])

        for b in ax.spines:
            axes[a].spines[b].set_visible(False)

        #plt.show()
    fig.tight_layout()
    # plt.savefig(rf'C:\Users\chloe\DE4\Masters\Figures\compare_17_21.pdf', dpi =300)
    index += 1
    # ground_truth = np.array(np.reshape(mask[index], (224,224,1)))
    #ground_truth = masks[i]
    # prediction = np.array(np.reshape(result[i], (224,224,1)))
    # prediction = result[index]

# plt.show()

def sqrt_fit(x, a, b):
    return a* np.sqrt(x) + b

dc_array = np.array(dc_array)
print(dc_array)
labels = ['Training images: 300','Training Images: 500']
plt.figure(figsize=[3.6, 3.6])
for i in range(2):
    popt, pcov = curve_fit(sqrt_fit, plant_pixels, dc_array[:,i])
    x = np.linspace(0,max(plant_pixels), 100)

    plt.plot(x ,sqrt_fit(x, *popt))
    plt.scatter(plant_pixels,dc_array[:,i], s=9, label = labels[i])

    print(np.mean(dc_array[:,i]))
plt.ylabel('Dice coefficient')
plt.xlabel('Image plant area')
plt.legend()
plt.savefig(rf'C:\Users\chloe\DE4\Masters\Figures\dc_area_21_23_plot.pdf', bbox_inches='tight', dpi =300)
plt.show()



# print(save_dict)
# print(len(save_dict))
# print(sum(save_dict.values())/len(save_dict))
# with open(rf'C:\Users\chloe\DE4\Masters\Models\Models_compare_small.json', 'w') as f:
#     json.dump(save_dict, f)
            

    