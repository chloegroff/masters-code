import os

image_folder = "/Volumes/MASTERS/03_15/"

for i, filename in enumerate(os.listdir(image_folder), start=1):
    if filename.split('_')[0] != '.':
        # if len(filename.split('_')) == 4:
        if filename.endswith('.jpg'):
            month, day, time, box_jpg = filename.split('_')
            boxNum = box_jpg.split('.')[0]

            hours, minutes, seconds = time.split(':')

            os.rename(image_folder + filename, image_folder + month + "_" + day + "_" + hours + "_" + minutes + "_" + seconds + "_" + boxNum + ".jpg")
        # if len(filename.split('_')) == 5:
        if filename.endswith('.tif'):
            month, day, time, boxNum, s_jpg = filename.split('_')

            hours, minutes, seconds = time.split(':')

            os.rename(image_folder + filename, image_folder + month + "_" + day + "_" + hours + "_" + minutes + "_" + seconds + "_" + boxNum + "_s.tif")
