import os

image_folder = "/Volumes/MASTERS/03_15/"

for i, filename in enumerate(os.listdir(image_folder), start=1):
    if filename.split('_')[0] != '.':
        # if len(filename.split('_')) == 4:
        if filename.endswith('.jpg'):
            month, day, time, box_jpg = filename.split('_')
            # month = filename.split('_')[0]
            # day = filename.split('_')[1]
            # time = filename.split('_')[2]

            boxNum = box_jpg.split('.')[0]

            hours, minutes, seconds = time.split(':')

            # hours = time.split(':')[0]
            # minutes = time.split(':')[1]
            # seconds = time.split(':')[2]

            os.rename(image_folder + filename, image_folder + month + "_" + day + "_" + hours + "_" + minutes + "_" + seconds + "_" + boxNum + ".jpg")
        # if len(filename.split('_')) == 5:
        if filename.endswith('.tif'):
            month, day, time, boxNum, s_jpg = filename.split('_')

            # month = filename.split('_')[0]
            # day = filename.split('_')[1]
            # time = filename.split('_')[2]
            # boxNum = filename.split('_')[3]

            hours, minutes, seconds = time.split(':')
            # hours = time.split(':')[0]
            # minutes = time.split(':')[1]
            # seconds = time.split(':')[2]

            os.rename(image_folder + filename, image_folder + month + "_" + day + "_" + hours + "_" + minutes + "_" + seconds + "_" + boxNum + "_s.tif")
