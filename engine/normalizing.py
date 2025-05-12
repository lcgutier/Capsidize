############################################################################### Normalizing_Cryo_Images Notebook ############################################################################
# print("Starting the normalizing.py script...")
import numpy as np
from PIL import Image
import sys

# for joining files paths for tif images
import os

# for removing the background
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball

# for displaying the images
import tempfile

"""
command line args: img_path, full_empty
"""

"""Note: the normalization code has been changed since normalizing this data. The images 
are now normalized by a slightly different process. 

The 100% full cryo images were normalized as:
1) Rolling Ball 200
2) CLAHE: clipsize= 2.0, tilesize= 8.0

The 100% empty cryo images were normalized as:
1) CLAHE: clipsize= 2.0, tilesize= 8.0
2) Rolling Ball 200
3) CLAHE: clipsize= 2.0, tilesize= 100"""

######################## Resize the images to 3 channels ########################################

temp_dir = tempfile.mkdtemp()

img_paths = sys.argv[1:]
processed_image_paths = []
sum_image_areas = 0

for img_path in img_paths:

    # Open the image and convert it to an array
    img_rgb = Image.open(img_path)
    img_rgb = np.array(img_rgb).astype(np.uint8)
    
    # get the area of the image
    area = img_rgb.shape[0] * img_rgb.shape[1]
    sum_image_areas += area

    # Duplicate the first channel into 3 channels
    img_rgb = np.repeat(img_rgb[:, :, np.newaxis], 3, axis=2)
    
    lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    l = np.array(l, dtype=np.uint8)

    ############################# Apply CLAHE once to the image ########################################

    # Apply CLAHE to the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe1_img = clahe.apply(l)
    
    ############################# Apply Rolling Ball to the image ########################################
    radius = 200
    rolling_ball_img, background = subtract_background_rolling_ball(clahe1_img, radius, light_background=True, use_paraboloid=False,
                                                             do_presmooth=True)
    
    ############################# Apply CLAHE a second time to the image ########################################
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(100, 100))
    clahe2_img = clahe.apply(rolling_ball_img)

    # Convert the image to RGB    
    clahe_img_rgb = cv2.cvtColor(clahe2_img, cv2.COLOR_GRAY2RGB)
    
    # Save only the first channel
    clahe_img_rgb_1c = clahe_img_rgb[:, :, 0]

    ############################# Save the image as a jpg (electron doesn't like tif) ########################################

    # Split the base name and the extension
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Create the new filename with the .jpg extension
    new_filename = base_name + ".jpg"

    # Join the new filename with the temporary directory path
    output_path = os.path.join(temp_dir, new_filename)
    Image.fromarray(img_rgb).save(output_path)
    processed_image_paths.append(output_path)
    
print(f"sum_image_areas: {sum_image_areas}")
print(temp_dir)
    
# print("Processed Images Paths from python...:")
print("\n".join(processed_image_paths))
