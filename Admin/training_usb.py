import os
# import cv2
import shutil

usb_drive = "/Volumes/MASTERS/"
output_folder = "/Volumes/MASTERS/imagesMasks/"

#folders = ['02_16', '02_17', '02_19', '02_20', '02_22', '02_23', '02_28', '03_07', '03_15', 'STUDY2']
folders = ['02_17', '02_19']
#glob(image_folder/**.jpg)

for folder in folders:
    for i, filename in enumerate(os.listdir(usb_drive + folder), start=1):
        if filename.endswith('.jpg'):
            remove_extension = filename.split('.')[0]
            maskname = remove_extension + '_s.jpg'
            if os.path.exists(usb_drive + folder + '/' + maskname):
                shutil.copy(usb_drive + folder + '/' + filename, output_folder + filename)
                shutil.copy(usb_drive + folder + '/' + maskname, output_folder + maskname)
                # cv2.imwrite(output_folder, filename)
                # cv2.imwrite(output_folder, maskname)
