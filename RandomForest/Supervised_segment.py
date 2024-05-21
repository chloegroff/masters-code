import numpy as np
from skimage import future, feature
from sklearn.ensemble import RandomForestClassifier
import cv2
import pickle as pkl
from os.path import sep
import numpy.typing as npt
import matplotlib.pyplot as plt
from functools import partial
from glob import glob as glob


def manual_segment(save_root: str, images_root: str = None, files: str = None):
    to_mask = files
    
    if files is None:
        files = glob(images_root + sep + '*.jpg')
        masks = glob(images_root + sep + '*_s*')
        mask_stamps = [i[:-6] for i in masks]

        to_mask = []
        for i in files:
            if i[:-4] not in mask_stamps:
                to_mask.append(i)
        

    for i in to_mask:
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = future.manual_lasso_segmentation(img,alpha=1,return_all=True)

        plt.imshow(mask)
        plt.show()
        cv2.imwrite(save_root + sep + i[:-4] + '_s.tif', img)


def predict_mask(
        model: RandomForestClassifier, files: list[str], 
        feature_func: callable, save_root: str) -> None:

    for i in files:
        img = cv2.imread(i)

        #img = cv2.cvtColor(img, cv2.COLOR_BRG2HLS)

        feats = feature_func(img)

        mask = future.predict_segmenter(feats, model) 
        cv2.imwrite(save_root + sep + i[:-3] + '_model.tif', mask)


def stich_images(files: list[str], mask = False) -> npt.NDArray: 
    flag = None
    if mask:
        flag = 0
    for i, f in enumerate(files):
        if i == 0:
            long_img = cv2.imread(f, flag)
        
        else:
            nimg = cv2.imread(f, flag)

            long_img = np.concatenate([long_img, nimg], axis = 0)

    if mask:
        long_img = np.where(long_img > 1, 2,1)
        long_img = np.array(long_img, int)
        return long_img

    return cv2.cvtColor(long_img, cv2.COLOR_BGR2HLS)

def rand_unique(num, files) -> list[int]:

    arr = np.zeros((num), dtype=int)

    for idx, i in enumerate(arr):
        while arr[idx] in np.delete(arr, idx):
            arr[idx] = np.random.randint(0, len(files))
    
    return arr

sigma_min = 1
sigma_max = 50

features_func = partial(
    feature.multiscale_basic_features,
    intensity=True,
    edges=True,
    texture=False,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
    channel_axis=-1,
    num_sigma = 2,
)


if __name__ == '__main__':



    images = np.array(glob(r'Images_Masks\Louis\Louis' + sep + '*.jpg'))
    masks = np.array(glob(r'Images_Masks\Louis\Louis' + sep + '*_s*')) 


    args = rand_unique(8, images)

    long_img = stich_images(images[args])
    long_mask = stich_images(masks[args], mask= True)


    # fig, ax = plt.subplots(nrows = 1, ncols = 2)
    # ax[0].imshow(long_img)
    # ax[1].imshow(long_mask)
    # plt.show()




    features = features_func(long_img)

    print('all good')
    # for i in range(features.shape[2]):
    #     plt.imshow(features[:,:,i])
    #     plt.show()
    clf = RandomForestClassifier(n_jobs=-1)
    clf = future.fit_segmenter(long_mask, features, clf)
    result = future.predict_segmenter(features, clf)


    with open('Trained_Classifier.pkl', 'wb') as f:
        pkl.dump(clf, f)

    fg_img = cv2.imread('Full Growth.jpg')
    #fg_img=cv2.cvtColor(fg_img,cv2.COLOR_BGR2HLS)

    fg_feats = features_func(fg_img)
    fg_result =  future.predict_segmenter(fg_feats, clf)

    sg_img = cv2.imread('Smol Growth.jpg')
    #sg_img=cv2.cvtColor(sg_img,cv2.COLOR_BGR2HLS)

    sg_feats = features_func(sg_img)
    sg_result =  future.predict_segmenter(sg_feats, clf)

    fig, ax = plt.subplots(2, 2)

    ax[0,0].imshow(sg_result)
    ax[0,1].imshow(fg_result)

    ax[1,0].imshow(cv2.cvtColor(sg_img, cv2.COLOR_HLS2RGB))
    ax[1,1].imshow(cv2.cvtColor(fg_img, cv2.COLOR_HLS2RGB))
    plt.show()