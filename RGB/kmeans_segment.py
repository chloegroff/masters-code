import numpy as np
import umap
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('Full Growth.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:,:,1])
plt.show()
newimg = np.zeros_like(img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if 22 < img[i,j,0] < 45:
            newimg[i,j] = img[i,j]

plt.imshow(newimg)
plt.show()
img = cv2.cvtColor(newimg, cv2.COLOR_HLS2RGB)
img = cv2.cvtColor(newimg, cv2.COLOR_RGB2LAB)
vectorized = img.reshape((-1,3))

img.shape
print(img.shape)
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)

res = center[label.flatten()]
result_image = res.reshape((img.shape))

figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()

# edges = cv2.Canny(img,40,100)
# plt.figure(figsize=(figure_size,figure_size))
# plt.subplot(1,2,1),plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,2,2),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

