import os
import cv2

masters_folder = "/Users/chloegroff/Documents/DE4/Masters/Pi_Pictures/ChloeMasters/"
output_folder = "/Volumes/MASTERS/imagesMasks/"

#glob(image_folder/**.jpg)

for folder in os.listdir(masters_folder):
    for i, filename in enumerate(os.listdir(masters_folder + folder), start=1):
        if filename.endswith('.jpg'):
            name = filename.split('.')[0]
            maskname = name + '_s.jpg'
            if os.path.exists(masters_folder + folder + '/' + maskname):
                cv2.imwrite(output_folder, filename)
                cv2.imwrite(output_folder, maskname)
