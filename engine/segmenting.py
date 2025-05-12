############################################################# Import Libraries #################################################################
print("script message: starting segmenting.py")

from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
# import supervision as sv

# for joining files paths for tif images
import os

# piecewise linear fit
import numpy as np

# for removing the background
from cv2_rolling_ball import subtract_background_rolling_ball

# Converting the data to a COCO JSON file
import json

# for displaying the images
import tempfile

import sys
import copy

#saving the sam results
import pickle


print("script message: loaded all libraries")

############################################################# Define Input Paths ##############################################################

temp_dir = tempfile.mkdtemp()
img_paths = sys.argv[1:]

annotated_image_paths = []

##################################################### Load in Sam Model And Segment Images ####################################################

# Determine the model type as well as the checkpoint. Include the path to the checkpoint that you downloaded.
sam = sam_model_registry["vit_h"](checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam_models/sam_vit_h_4b8939.pth"))
# sam = sam_model_registry["vit_l"](checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam_models/sam_vit_l_0b3195.pth"))
# sam = sam_model_registry["vit_b"](checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam_models/sam_vit_b_01ec64.pth"))

# print("script message: loading sam model")
# Set the predictor
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

#################### Begining of tutorial code ####################
### Encoding code here can be found at:
# https://stackoverflow.com/questions/49494337/encode-numpy-array-using-uncompressed-rle-for-coco-dataset

def encode(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}

    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    # note that the odd counts are always the numbers of zeros
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))

    rle["counts"] = lengths.tolist()

    return rle
#################### End of tutorial code ####################

# Get the sam results for all of the cryo images

sam_results_combined = []
images_dict = []
img_rgbs = []

sam_results_by_image = []

# # print("script message: getting sam results")
# # get each of the images from the img_path one at a time and perform the sam mask generation
for i,img_path in enumerate(img_paths):
    print(f"script message: getting sam results for image {i}")
    
    img_rgb = Image.open(img_path)
    
    img_rgb = np.array(img_rgb).astype(np.uint8)
    # img_rgb = np.repeat(img_rgb[:, :, np.newaxis], 3, axis=2)
    img_rgbs.append(img_rgb)
    print("script message: img_rgb shape:", img_rgb.shape)
    
    sam_result = mask_generator.generate(img_rgb)
    
    ################################################# Convert Sam Annotations to COCO JSON ##################################################
    
    sam_results_by_image.append(copy.deepcopy(sam_result))
    
    for j,annotation in enumerate(sam_result):
            # annotation_id
            annotation["id"] = j+1
            
            # segmentation
            # convert the "segmentation" value to binary
            binary_mask = (annotation["segmentation"]).astype(np.uint8)

            rle_mask = encode(binary_mask)
            
            # bbox and area are already included
            
            # replace the "segmentation" value with the rle mask
            annotation["segmentation"] = rle_mask
            
            # iscrowd
            annotation["iscrowd"] = 1
            
            # attributes
            annotation["attributes"] = {}
            annotation["attributes"]["occluded"] = "false" 
            
            # image_id
            annotation["image_id"] = i+1
            
            # category_id
            annotation["category_id"] = 1
            
            # remove point_coords, stability_score, crop_box, predicted_iou
            annotation.pop("point_coords")
            annotation.pop("stability_score")
            annotation.pop("crop_box")
            annotation.pop("predicted_iou")
            
    for result in sam_result:
        sam_results_combined.append(result)
        
    # get the image name from the path
    img_name = os.path.basename(img_path)    
    
    image_dict = {}
    image_dict["id"] = i+1
    width, height = Image.open(img_path).size
    image_dict["width"] = width
    image_dict["height"] = height
    image_dict["file_name"] = img_name
    image_dict["license"] = 0
    image_dict["flickr_url"] = ""
    image_dict["coco_url"] = ""
    image_dict["date_captured"] = 0
    
    images_dict.append(image_dict)

# licenses, info, categories
info_dict = {"licenses":[{"name":"","id":0,"url":""}],
             "info":{"contributor":"","date_created":"","description":"","url":"","version":"","year":""},
             "categories":[{"id":1,"name":"Full","supercategory":""},{"id":2,"name":"Partial","supercategory":""},{"id":3,"name":"Empty","supercategory":""},
                           {"id":4,"name":"Aggregation","supercategory":""}, {"id":5,"name":"Ice","supercategory":""}, {"id":6,"name":"Broken","supercategory":""},
                           {"id":7,"name":"Background","supercategory":""}]}

# Combine the dicts:
coco_json = dict(info_dict, images = images_dict, annotations = sam_results_combined)

json_output_path = os.path.join(temp_dir, "cryo_sam_results.json")
sam_results_by_image_output_path = os.path.join(temp_dir, "sam_results_by_image.pickle")

print(json_output_path)
print(sam_results_by_image_output_path)

# print("script message: saving coco json")
# Save the coco_json to a json file
with open(json_output_path, 'w') as f:
    json.dump(coco_json, f)
    
# Save the sam_results_by_image to a pickle file
with open(sam_results_by_image_output_path, 'wb') as f:
    pickle.dump(sam_results_by_image, f)
    
print("\n".join(annotated_image_paths))
    