import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the path to the folder containing the microscoped plant
# image_folder = "D:\William\Design_Engineering\Year_3\Research Placement\growth images\growth images cam1"
# image_folder = "D:\William\Design_Engineering\Year_3\Research Placement\growth images\growth images cam2"
# image_folder = "/Users/chloegroff/Documents/DE4/Masters/arabidopsis_soil/"
image_folder = "/Users/chloegroff/Documents/DE4/Masters/Pi_Pictures/ChloeMasters/Segment_test"

# Specify the folder to save the segmented images
#output_folder = "/Users/chloegroff/Documents/DE4/Masters/arabidopsis_soil/segmented/"
output_folder = "/Users/chloegroff/Documents/DE4/Masters/Pi_Pictures/ChloeMasters/Segment_test_s"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize lists to store the timeline and green pixel counts
green_pixel_counts = []

# Iterate over each image in the folder
for i, filename in enumerate(os.listdir(image_folder), start=1):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image = cv2.imread(os.path.join(image_folder, filename))

        # Convert the image to the RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Define the lower and upper threshold values for green pixels
        lower_green = (30, 29, 12)
        upper_green = (120, 120, 100)

        # Define a string of the threshold avlues
        lower_string = ''.join([str(lower_green[0]), '_', str(lower_green[1]), '_', str(lower_green[2])])
        upper_string = ''.join([str(upper_green[0]), '_', str(upper_green[1]), '_', str(upper_green[2])])

        # Create a binary mask of the green pixels
        mask = cv2.inRange(image_rgb, lower_green, upper_green)

        # Apply the mask to the original image
        segmented_image = cv2.bitwise_and(image, image, mask=mask)

        # Save the segmented image with the original filename appended with "_s"
        output_filename = os.path.splitext(filename)[0] + '_' + lower_string + '_' + upper_string + "_s" + os.path.splitext(filename)[1]
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, segmented_image)

        # Count the number of green pixels
        green_pixel_count = cv2.countNonZero(mask)

        # Add the data to the timeline and green pixel count lists
        green_pixel_counts.append(green_pixel_count)

#Add green pixels found in every image together
total_pixels = np.sum(green_pixel_counts)
area = total_pixels * 0.00703e-3**2

print("Total pixels: ", total_pixels)
print("Total area: ", area, "m^2")
