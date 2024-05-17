from tensorflow import keras

import matplotlib
import matplotlib.image as Image
import matplotlib.pyplot as plt
import glob
import numpy as np

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

loaded_model = keras.models.load_model(model)
result = loaded_model.predict(images)
result = result > 0.5

axes[0].imshow(images[0])
axes[0].set_xlabel('Raw Image')
axes[1].imshow(np.reshape(masks[0]*255, (dim, dim)), cmap="gray")
axes[1].set_xlabel('Ground Truth')
axes[2].imshow(np.reshape(result[0]*255, (dim, dim)), cmap="gray")
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
        
    plt.savefig(rf'C:\Users\chloe\DE4\Masters\Models\Model_{model_num}_example_1.pdf', dpi =300)