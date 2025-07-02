import os
import cv2 as cv
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from wiener_filter import WienerFilter

## Path to Image folder
image_folder = "HAM_10000_Dataset/HAM10000_images_part_1"
## List of Images
images = []

for filename in os.listdir(image_folder)[:50]:
    if filename.endswith(".jpg"):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path) 
        images.append(img)

## Grayscale Images
gray_scale_images = []

for image in images:
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
    gray_scale_images.append(gray_image)

filtered_images = []

for image in gray_scale_images:
    image_with_filter = WienerFilter(image, (5, 5))
    output_image_with_filter = image_with_filter.estimateOutput()
    sharpen_kernel = np.array([[0,-1,0], [-1,7,-1], [0,-1,0]])
    sharpened_image_with_filter = cv.filter2D(output_image_with_filter, -1, sharpen_kernel)
    filtered_images.append(sharpened_image_with_filter)
