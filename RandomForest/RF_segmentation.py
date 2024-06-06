import numpy as np
from skimage import future, feature
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import cv2
import pickle as pkl
import json
import random
from os import remove
from os.path import sep
import numpy.typing as npt
import matplotlib.pyplot as plt
from functools import partial
from glob import glob as glob


def open_img(file):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        return img

def open_mask(file):
        img = cv2.imread(file, 0)
        img = np.where(img > 1, 2,1)
        img = np.array(img, int)
        return img

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

        img = cv2.cvtColor(img, cv2.COLOR_BRG2HLS)

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

def rand_unique(num, files1, files2) -> list[int]:
    files = list(zip(files1,files2))
    random.shuffle(files)
    files = np.array(files)[:num]
    return files


def mean_iou(y_true, y_pred):

    pred_sum = np.sum(y_pred)
    gt_sum = np.sum(y_true)
    intersection = np.sum(y_true @ y_pred)
    
    union = pred_sum + gt_sum - intersection

    iou = intersection/union
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return np.mean(2. * intersection + smooth) / (union + smooth)



def mean_metric(metric: callable, img_files: list[str], mask_files: list[str], 
                   model: RandomForestClassifier):
    
    mets = []
    for i, (img, mask) in enumerate(zip(img_files, mask_files)):
        img = open_img(img)
        mask = open_mask(mask)

        feats = features_func(img)

        pred = future.predict_segmenter(feats, model)
        met = metric(mask.flatten() - 1,pred.flatten() - 1)
        mets.append(met)

    if len(met.shape) > 1:
        return np.mean(mets,0)
    
    return np.mean(mets)

    



def grid_search(estimator_range, depth_range, features, train_mask, test_images, test_masks, save_path, name, metrics = [dice_coef, confusion_matrix]):

    prev_files = glob(save_path + sep + '*')
    for f in prev_files:
        remove(f)

    met_list = [[] for k in metrics]
    for i in tqdm(estimator_range):
        rows = [[] for k in metrics]
        for j in tqdm(depth_range):
            clf = RandomForestClassifier(n_estimators= i, max_depth = j, n_jobs=-1)
            clf = future.fit_segmenter(train_mask,features, clf)

            with open(save_path + sep + 'model_' + name + f'_nest_{i}_mdep_{j}.pkl', 'wb') as f:
                pkl.dump(clf, f)

            save_dict = {}            

            for k, mfunc in enumerate(metrics):
                met = mean_metric(mfunc, test_images, test_masks, clf)
                
                if not isinstance(met,float):
                    met = met.tolist()
                save_dict[mfunc.__name__] = met
                rows[k].append(met)
            

            
            with open(save_path + sep + 'meta_' + name + f'_nest_{i}_mdep_{j}.json', 'w') as f:
                json.dump(save_dict, f)

        for k,_ in enumerate(metrics):
            met_list[k].append(rows[k])
    
    met_dict_a = {}
    met_dict_b = {}
    for k, mfunc in enumerate(metrics):
        met_dict_a[mfunc.__name__] = np.array(met_list[k])
        met_dict_b[mfunc.__name__] = met_list[k]

    return met_dict_a, met_dict_b
        



            

            


            



sigma_min = 1
sigma_max = 10

features_func = partial(
    feature.multiscale_basic_features,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
    channel_axis=-1,
    num_sigma = 3,
)


if __name__ == '__main__':



    images = np.array(glob(r'C:\Users\Pouis\Documents\Uni Shit\Chloe Data\Training' + sep + '*_i*'))
    masks = np.array(glob(r'C:\Users\Pouis\Documents\Uni Shit\Chloe Data\Training' + sep + '*_s*')) 


    files = rand_unique(30, images, masks)

    long_img = stich_images(files[:,0])
    long_mask = stich_images(files[:,1], mask= True)


    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    ax[0].imshow(long_img)
    ax[1].imshow(long_mask)
    plt.show()




    features = features_func(long_img)

    print('all good')
    # for i in range(features.shape[2]):
    #     plt.imshow(features[:,:,i])
    #     plt.show()
    clf = RandomForestClassifier(n_jobs=-1)
    clf = future.fit_segmenter(long_mask, features, clf)
    result = future.predict_segmenter(features, clf)

    with open('RF_50_HLS_del8.pkl', 'wb') as f:
        pkl.dump(clf, f)

    with open('RF_50_HLS_del8.pkl', 'rb') as f:
        clf = pkl.load(f)

    fg_img = cv2.imread(r"C:\Users\Pouis\Documents\Uni Shit\Chloe Data\Validation\0_03_18_13_00_55_8_i.tif")
    fg_img=cv2.cvtColor(fg_img,cv2.COLOR_BGR2HLS)

    fg_feats = features_func(fg_img)
    fg_result =  future.predict_segmenter(fg_feats, clf)

    sg_img = cv2.imread(r"C:\Users\Pouis\Documents\Uni Shit\Chloe Data\Validation\8_02_20_14_15_41_8_i.tif")
    sg_img=cv2.cvtColor(sg_img,cv2.COLOR_BGR2HLS)

    sg_feats = features_func(sg_img)
    sg_result =  future.predict_segmenter(sg_feats, clf)

    fig, ax = plt.subplots(2, 2)

    ax[0,0].imshow(sg_result)
    ax[0,1].imshow(fg_result)

    ax[1,0].imshow(cv2.cvtColor(sg_img, cv2.COLOR_HLS2RGB))
    ax[1,1].imshow(cv2.cvtColor(fg_img, cv2.COLOR_HLS2RGB))
    plt.show()