############################################################# Import Libraries #################################################################
print("script message: starting patching.py")

import json
import cv2
import numpy as np
import pandas as pd
import sys
import tempfile

print("script message: loaded all libraries")

####################################################### Define Input and Output Paths ##########################################################

# in files and variables
coco_json_path = sys.argv[1]
image_folder_data = sys.argv[2]
background = True # Choose a background type. True for keeping the background, False for removing the background

# out files
temp_dir = tempfile.mkdtemp()

print("script message: defined input and output paths")
# print(coco_json_path)
# print(temp_dir)
############################################################### Create Patches #################################################################

#################### Begining of tutorial code ####################
# The decode_fast function is referenced from the following link:
# https://gist.github.com/akTwelve/dc0bbbf26fb14493898fc74cd2aa7f74

def decode_fast(mask_rle, shape):
    # get the width and height of the image
    height, width = shape
    zero_one = np.zeros_like(mask_rle, dtype=np.uint8)
    zero_one[1::2] = 255
    zero_one = zero_one.reshape((len(mask_rle), 1))
    expanded = np.repeat(zero_one, mask_rle, axis=0)
    filled = np.append(expanded, np.zeros(width * height - len(expanded)))
    im_arr = filled.reshape((height, width), order='F').astype(np.uint8)    
    return im_arr

####################### End of tutorial code #######################

# Load in the JSON files. Create a for loop that loads in the json files 
with open(coco_json_path) as f:
    data = json.load(f)

# initialize lists to store patch names and labels
patch_names = []
labels = []

for annotation in data['annotations']:
    # print(f"script message: Processing patches for annotation {annotation['id']}")
    image_id = annotation['image_id']
    image_info = next((img for img in data['images'] if img['id'] == image_id), None) #takes only the image info for the image id of that annotation

    if image_info:
        image_name = image_info['file_name']

        image_file = f"{image_folder_data}/{image_name}"
        original_image = cv2.imread(image_file)

        ########################## Create Patches Using BBOX ###########################
        if background == True: # if true, here we keep the background
            bbox = annotation['bbox']
            x, y, width, height = map(int, bbox)
            patch = original_image[y:y+height, x:x+width]

        ##################### Create Patches Using Segmentation ########################
        else: # if false, here we remove the background
            counts_size = annotation['segmentation']
            mask_rle = annotation['segmentation']['counts']
            
            # Convert the mask_rle to a numpy array
            mask_rle = np.array(mask_rle)
            
            mask = decode_fast(mask_rle, (image_info['height'], image_info['width']))
            
            # Apply the mask to the original image
            patch = cv2.bitwise_and(original_image, original_image, mask=mask)
            
            # Use the bbox to crop the patch
            bbox = annotation['bbox']
            x, y, width, height = map(int, bbox)
            patch = patch[y:y+height, x:x+width]
        
        # get the label of the annotation
        label = annotation['category_id']
        labels.append(label)
        
        # Save the patch
        patch_name = f"patch_{image_id}_{annotation['id']}.png"
        patch_filename = f"{temp_dir}/{patch_name}"
        
        # save the patch name to a list of patch names
        patch_names.append(patch_name)
        
        # Save the patch
        cv2.imwrite(patch_filename, patch)
            
# When done with the annotations for one dataset, print a message
print(f"script message: Finished processing patches")
            
# Create a df of the patch names and labels
names_labels_df = pd.DataFrame({'patch_name': patch_names, 'label': labels})

############################################################### Save Patches ####################################################################

print(temp_dir)

# Save the data to predicting_data
names_labels_df.to_csv(f"{temp_dir}/names_labels_df.csv", index=False)