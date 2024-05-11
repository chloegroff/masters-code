import os

image_folder = "/Users/chloegroff/Documents/DE4/Masters/Pi_Pictures/ChloeMasters/STUDY2/"

for i, filename in enumerate(os.listdir(image_folder), start=1):
    if filename.endswith('.jpg'):
        dayMonth = filename.split('_')[0]
        time = filename.split('_')[1] + '_' + filename.split('_')[2] + '_' + filename.split('_')[3]
        boxNum = filename.split('_')[4].split('.')[0]

        day = ''.join(filter(str.isnumeric, dayMonth))
        month = ''.join(filter(str.isalpha, dayMonth))

        if month == "February":
            monthDigit = "02"
        elif month == "March":
            monthDigit = "03"
        elif month == "April":
            monthDigit = "04"

        if not boxNum.isdigit():
            boxNum = ''.join(filter(str.isnumeric, boxNum))

        #print(monthDigit + "_" + day + "_" + boxNumber + ".jpg")
        os.rename(image_folder + filename, image_folder + monthDigit + "_" + day + "_" + time + "_" + boxNum + ".jpg")
