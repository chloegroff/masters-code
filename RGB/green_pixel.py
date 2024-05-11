import cv2
import matplotlib.pyplot as plt
import os

# Define the path to the folder containing the time-lapse images
# image_folder = "D:\William\Design_Engineering\Year_3\Research Placement\growth images\growth images cam1"
# image_folder = "D:\William\Design_Engineering\Year_3\Research Placement\growth images\growth images cam2"
image_folder = "documents/DE4/Masters/Box8_Microscope"


# Initialize lists to store the timeline and green pixel counts
timeline = []
green_pixel_counts = []

# Iterate over each image in the folder
for i, filename in enumerate(os.listdir(image_folder), start=1):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image = cv2.imread(os.path.join(image_folder, filename))

        # Convert the image to the RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Define the lower and upper threshold values for green pixels
        lower_green = (90, 95, 48)
        upper_green = (132, 139, 100)

        # Create a binary mask of the green pixels
        mask = cv2.inRange(image_rgb, lower_green, upper_green)

        # Count the number of green pixels
        green_pixel_count = cv2.countNonZero(mask)

        # Add the data to the timeline and green pixel count lists
        timeline.append(i)
        green_pixel_counts.append(green_pixel_count)

# Plot the timeline against the number of green pixels
plt.plot(timeline, green_pixel_counts)
plt.xlabel('Timeline')
plt.ylabel('Number of Green Pixels')
plt.title('Plant Growth Progress')
plt.xticks(rotation=45)
plt.show()
