from tensorflow import keras

import matplotlib
import matplotlib.image as Image
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2

fig, axes = plt.subplots(ncols = 3, )

fig.set_figwidth(3.6)
fig.set_figheight(2)

model_num = 1
model = rf'C:\Users\chloe\DE4\Masters\Models\Model_{model_num}.keras'

dim = 224
val_image = r'C:\Users\chloe\DE4\Masters\Dataset\validation_imgs'
val_image_save = rf'C:\Users\chloe\DE4\Masters\Models\Model_{model_num}_example.pdf'

images = glob.glob(val_image + r'\*_i.tif')
masks = glob.glob(val_image + r'\*_s.tif')

img = cv2.imread(images[0])

loaded_model = keras.models.load_model(model, compile=False)
result = loaded_model.predict(img)
result = result > 0.5

# model = load_model("aaaa.h5", compile=False)
# model.compile(loss=custom_loss, optimizer='adam', metrics=custom_loss)
# model.fit(...)

#for i in range(len(images) - 1):
for i in range(1):
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
            
        plt.savefig(rf'C:\Users\chloe\DE4\Masters\Models\Model_{model_num}_example_{i}.pdf', dpi =300)