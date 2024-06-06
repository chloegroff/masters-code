import signal
import sys
import RPi.GPIO as GPIO
import time
import tkinter as tk
import os
from picamera import PiCamera

BUTTON_GPIO = 16 
camera = PiCamera()
camera.resolution = (3280,2464) #Adjust camera resolution

# Define the time interval between images (in seconds)
time_interval = 4  # time interval for every 30 minutes
long_time_interval = 30
# Define the number of images to capture
num_images = 8  # 48 images everyday for 60 days time span

# Define the number of image cycles
num_cycles = 2

# Set the directory where the images will be saved
save_directory = "/home/pi/Desktop/ChloeMasters/15_March"

def take_picture():
	# Time for printer to set up and begin the initialisation process
    #print(f"Printer setting up test")
    #time.sleep(13)
    # Time for printer to initialise 130cm out of the way
    #print(f"Printer initialising")
    #time.sleep(43.5)
    print(f"Test start")
    # Time for printer to move onto first plant
    time.sleep(43)
    # Begin image loop
    for i in range(4):
        # Generate a unique filename based on current time for each image
        current_time = time.strftime("%d%B_%H:%M:%S")
        plantnumber = i+1
        image_filename = f"{current_time}_Box{plantnumber}.jpg"
        
        # Capture the start time
        start_time = time.time()
    
        # Capture and save the image
        camera.capture(f"{save_directory}/{image_filename}")
        
        # Calculate the time taken for image capture
        capture_time = time.time() - start_time

        print(f"Captured {image_filename}")
        # Print the time taken
        print(f"Time taken for {image_filename}: {capture_time:.2f} seconds")
        
        new_time_interval = time_interval - capture_time
    
        # Wait for the specified time interval
        time.sleep(new_time_interval)
    time.sleep(31)
    for i in range(4):
        # Generate a unique filename based on current time for each image
        current_time = time.strftime("%d%B_%H:%M:%S")
        plantnumber = i+5
        image_filename = f"{current_time}_Box{plantnumber}.jpg"
        
        # Capture the start time
        start_time = time.time()
    
        # Capture and save the image
        camera.capture(f"{save_directory}/{image_filename}")
        
        # Calculate the time taken for image capture
        capture_time = time.time() - start_time

        print(f"Captured {image_filename}")
        # Print the time taken
        print(f"Time taken for {image_filename}: {capture_time:.2f} seconds")
        
        new_time_interval = time_interval - capture_time
    
        # Wait for the specified time interval
        time.sleep(new_time_interval)
        # NOTE: 2 second delay before printing following statement:
    print(f"Test done")
    # Printer moves out of way of plants
    print(f"Printer initialising")
    # 74s instead of 76s because there is a 2 second delay between last picture and processing this command
    time.sleep(80)
    # Time between cycles
    print(f"Wait 2 hours")
    #time.sleep(10)

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)


#GPIO.add_event_detect(BUTTON_GPIO, GPIO.RISING, callback=button_pressed_callback, bouncetime=500)
print("Waiting...")

while(True):
	if not GPIO.input(BUTTON_GPIO):
		take_picture()
		#print("button pressed")
	#sleep(0.001)
