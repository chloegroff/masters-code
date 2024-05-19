from tensorflow import keras
import tensorflow as tf
import matplotlib
import matplotlib.image as Image
from os.path import sep
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import zoom
import numpy as np
import cv2
import numpy.typing as npt
import json
import os
import string


def dice_coef(img, img2):
        if img.shape != img2.shape:
            raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
        else:
            
            lenIntersection=0
            
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if ( np.array_equal(img[i][j],img2[i][j]) ):
                        lenIntersection+=1
             
            lenimg=img.shape[0]*img.shape[1]
            lenimg2=img2.shape[0]*img2.shape[1]  
            value = (2. * lenIntersection  / (lenimg + lenimg2))
        return value

def get_imgs(img_files: str, hls: bool = True, mask: bool = False, resolution: list[int,int] = [224,224]) -> npt.NDArray:
    imgs = []
    for i, img_file in enumerate(img_files):

        if mask:
            img = cv2.imread(img_file, 0)    

        else:
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if hls and not mask:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        scales = [resolution[0]/(img.shape[0]),resolution[1]/(img.shape[1]), 1]
        if mask:
            
            img = zoom(img, [scales[0], scales[1]])
        
        else:
            img = zoom(img, scales)



        imgs.append(img)

    return np.array(imgs, dtype= np.uint32)

dim = 224
val_image = r'C:\Users\chloe\DE4\Masters\Dataset\validation_imgs'
image_files = glob.glob(val_image + sep +  '*_i.tif')
mask_files = glob.glob(val_image + sep + '*_s.tif')

# Define model numbers to compare
models = [8,9,10]
# Create a string of model numbers seperated by '_' e.g. '8_9_10'
models_str = ''
for i in range(len(models)-1):
    models_str = models_str + str(models[i]) + '_'
models.str = models_str + str(models[-1])
print(models_str)

predictions = []

images = get_imgs(image_files, hls = False)
plt.imshow(images[0])
plt.show()
masks = get_imgs(mask_files, mask = True)

# Loops through each model specified to generate a prediction for an image
for i, model in enumerate(models):
    model = rf'C:\Users\chloe\DE4\Masters\Models\Model_{model}.keras'
    loaded_model = keras.models.load_model(model, compile=False)

    result = loaded_model.predict(images / 255)
    result = result > 0.5
    predictions.append(result)

predictions = np.array(predictions)


matplotlib.rcParams.update({'font.size': 12, "font.family": "Times New Roman"})

# Loops through each validation image in folder
for k, mask in enumerate(masks):

    fig, axes = plt.subplots(ncols = 2, nrows = 2)
    fig.set_figwidth(3.6)
    fig.set_figheight(3.6)

    axes = axes.flatten()

    axes[0].imshow(np.reshape(mask*255, (dim, dim)), cmap="gray")
    axes[0].set_xlabel('Ground Truth')
    for j, result in enumerate(predictions[:,k]):
        axes[j + 1].imshow(np.reshape(result*255, (dim, dim)), cmap="gray")
        axes[j + 1].set_xlabel('Model ' + str(models[j]))



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

    plt.savefig(rf'C:\Users\chloe\DE4\Masters\Figures\Mask_{k}_compare_{models_str}.pdf', dpi =300)

    #ground_truth = np.array(np.reshape(mask[index], (224,224,1)))
    #ground_truth = masks[i]
    #prediction = np.array(np.reshape(result[i], (224,224,1)))
    # prediction = result[index]
    # image_name = os.path.relpath(image_files[index], val_image)
    # save_dict[image_name] = {
    #         'Dice coefficient': dice_coef(ground_truth, prediction)} 


# with open(rf'C:\Users\chloe\DE4\Masters\Models\Model_{models_str}compare.json', 'w') as f:
#     json.dump(save_dict, f)
            

    