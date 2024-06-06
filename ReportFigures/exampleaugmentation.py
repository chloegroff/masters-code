import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
from scipy.ndimage import zoom
import numpy as np
import numpy.typing as npt
import matplotlib

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

# path = rf'C:\Users\chloe\DE4\Masters\Dataset\Training_Data\50_03_17_06_15_03_4_i.tif'
# val_image_save = rf'C:\Users\chloe\DE4\Masters\Models\Model_{model_num}_example.pdf'
matplotlib.rcParams.update({'font.size': 12, "font.family": "Times New Roman", 
                            'figure.figsize': [7.2, 7.2], 'figure.dpi': 100, 'savefig.dpi': 300})
#image_files = glob.glob(val_image + sep +  '*_i.tif')
dim = 224
# image_files = glob.glob(path + sep +  '*_i.tif')
image_files = [r'C:\Users\chloe\DE4\Masters\Dataset\Training_Data_no8\71_03_11_09_19_22_4_i.tif', 
          r'C:\Users\chloe\DE4\Masters\Dataset\Training_Data_no8\344_03_11_09_19_22_4_i.tif', 
          r'C:\Users\chloe\DE4\Masters\Dataset\Training_Data_no8\299_03_11_09_19_22_4_i.tif', 
          r'C:\Users\chloe\DE4\Masters\Dataset\Training_Data_no8\443_03_11_09_19_22_4_i.tif']

images = get_imgs(image_files, hls = False, resolution=[dim,dim])

fig, axes = plt.subplots(ncols = 4)
fig.set_figwidth(7.2)
fig.set_figheight(4)
axes[0].imshow(images[0])
axes[0].set_xlabel('Raw Image')
axes[1].imshow(images[1])
axes[1].set_xlabel('(a)')
axes[2].imshow(images[2])
axes[2].set_xlabel('(b)')
axes[3].imshow(images[3])
axes[3].set_xlabel('(c)')

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
plt.savefig(rf'C:\Users\chloe\DE4\Masters\Figures\example_augmentation.pdf', bbox_inches='tight', dpi =300)
plt.imshow
plt.show()