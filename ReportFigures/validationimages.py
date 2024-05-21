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

def mean_iou_loss(img, img2):

    # pred_sum = keras.backend.sum(y_pred)
    # gt_sum = keras.backend.sum(y_true)
    # intersection = keras.backend.sum(tf.math.multiply(y_true,y_pred))
    
    # union = pred_sum + gt_sum - intersection

    # iou = intersection/union
    # return 1 - iou

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
        value = lenIntersection  / (lenimg + lenimg2)
    return value

# @keras.saving.register_keras_serializable()
# def dice_coef(y_true, y_pred, smooth=1):
#     y_pred = tf.cast(y_pred, tf.float32)
#     y_true = tf.cast(y_true, tf.float32)
#     intersection = keras.backend.sum(y_true * y_pred, axis=[1,2,3])
#     union = keras.backend.sum(y_true, axis=[1,2,3]) + keras.backend.sum(y_pred, axis=[1,2,3])
#     return keras.backend.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

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

# @keras.saving.register_keras_serializable()
def dice_loss(in_gt, in_pred):
    return 1 - dice_coef(in_gt, in_pred)

# @keras.saving.register_keras_serializable()
def true_positive_rate(y_true, y_pred):
    return keras.backend.sum(keras.backend.flatten(y_true)*keras.backend.flatten(keras.backend.round(y_pred)))/keras.backend.sum(y_true)

# @keras.saving.register_keras_serializable()
def dice_mean_iou(in_gt, in_pred):
    return (dice_loss(in_gt, in_pred) + mean_iou_loss(in_gt, in_pred)) / 2

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



model_num = 12
model = rf'C:\Users\chloe\DE4\Masters\Models\Model_{model_num}.keras'

dim = 224
val_image = r'C:\Users\chloe\DE4\Masters\Dataset\validation_imgs'
# val_image_save = rf'C:\Users\chloe\DE4\Masters\Models\Model_{model_num}_example.pdf'

#image_files = glob.glob(val_image + sep +  '*_i.tif')
image_files = glob.glob(val_image + sep +  '*_i.tif')
mask_files = glob.glob(val_image + sep + '*_s.tif')

#print(image_files)

# img = cv2.imread(images[0])

loaded_model = keras.models.load_model(model, compile=False)

images = get_imgs(image_files, hls = True)
plt.imshow(images[0])
plt.show()
masks = get_imgs(mask_files, mask = True)

result = loaded_model.predict(images / 255)
result = result > 0.5


# model = load_model("aaaa.h5", compile=False)
# model.compile(loss=custom_loss, optimizer='adam', metrics=custom_loss)
# model.fit(...)
index = 0
save_dict = {}
print(len(images))
#for i in range(len(images) - 1):
for i in range(len(image_files)):
    print(masks[i].shape)
    fig, axes = plt.subplots(ncols = 3, )
    fig.set_figwidth(3.6)
    fig.set_figheight(2)
    axes[0].imshow(images[i])
    axes[0].set_xlabel('Raw Image')
    axes[1].imshow(np.reshape(masks[i]*255, (dim, dim)), cmap="gray")
    axes[1].set_xlabel('Ground Truth')
    axes[2].imshow(np.reshape(result[i]*255, (dim, dim)), cmap="gray")
    axes[2].set_xlabel('Prediction')

    matplotlib.rcParams.update({'font.size': 12, "font.family": "Times New Roman"})

    for i, ax in enumerate(axes):
        axes[i].tick_params(
            axis='both',          # changes apply to the voth axis
            which='both', 
            left = False,   # both major and minor ticks are affected
            right = False,
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        
        axes[i].xaxis.set_ticklabels([])
        axes[i].yaxis.set_ticklabels([])

        for j in ax.spines:
            axes[i].spines[j].set_visible(False)

        #plt.show()
    plt.savefig(rf'C:\Users\chloe\DE4\Masters\Models\Model_{model_num}_val_{index + 1}.pdf', dpi =300)

    ground_truth = np.array(np.reshape(masks[index], (224,224,1)))
    #ground_truth = masks[i]
    #prediction = np.array(np.reshape(result[i], (224,224,1)))
    prediction = result[index]
    image_name = os.path.relpath(image_files[index], val_image)
    save_dict[image_name] = {
            'Dice coefficient': dice_coef(ground_truth, prediction)} 

    index += 1


with open(rf'C:\Users\chloe\DE4\Masters\Models\Model_{model_num}_val.json', 'w') as f:
    json.dump(save_dict, f)
            

    