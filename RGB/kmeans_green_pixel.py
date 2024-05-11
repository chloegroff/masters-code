import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Specify the folder containing the time-lapse images
image_folder = r'D:\William\Design_Engineering\Year_3\Research Placement\growth images\growth images cam2'

# Specify the folder to save the segmented images
output_folder = r'D:\William\Design_Engineering\Year_3\Research Placement\growth images\growth images cam2\segmented_images'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize lists to store the timeline and leaf area
timeline = []
leaf_areas = []

# Define the tolerance values for specific days
tolerance_dict = {
    8: 126,
    9: 126,
    10: 126,
    11: 133,
    12: 136,
    15: 125,
    16: 124,
    17: 123.5
}

# Iterate over each image in the folder
for i, filename in enumerate(os.listdir(image_folder), start=1):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Load the image
        image = cv2.imread(os.path.join(image_folder, filename))

        if image is None:
            continue

        # Convert the image to the RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Flatten the image to a 2D array of pixels
        pixels = image_rgb.reshape(-1, 3)

        # Apply K-means clustering to find dominant colors
        n_clusters = 5  # Adjust the number of clusters as per your image
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(pixels)

        # Get the RGB values of the cluster centers
        cluster_centers_rgb = kmeans.cluster_centers_.astype(int)

        # Find the cluster center with the closest green color
        green_center = min(cluster_centers_rgb, key=lambda x: abs(x[1] - x[0] - x[2]))

        # Set the tolerance value based on the day
        tolerance = tolerance_dict.get(i, 128)

        tolerance_array = np.array([tolerance, tolerance, tolerance])

        # Calculate the lower and upper threshold values
        lower_green = np.clip(green_center - tolerance_array, 0, 255)
        upper_green = np.clip(green_center + tolerance_array, 0, 255)

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a binary mask of the green areas using color thresholding
        mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Apply the mask to the original image
        segmented_image = cv2.bitwise_and(image, image, mask=mask)

        # Save the segmented image with the original filename appended with "_s"
        output_filename = os.path.splitext(filename)[0] + "_s" + os.path.splitext(filename)[1]
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, segmented_image)

        # Count the total green area in pixels
        leaf_area = cv2.countNonZero(mask)

        # Add the data to the timeline and leaf area lists
        timeline.append(i)
        leaf_areas.append(leaf_area)

# Plot the timeline against the total leaf area
plt.plot(timeline, leaf_areas)
plt.xlabel('Day')
plt.ylabel('Total Leaf Area (pixels)')
plt.title('Plant Growth Progress')
plt.show()
