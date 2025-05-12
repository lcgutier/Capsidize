############################################################# Import Libraries #################################################################
print("script message: starting updating.py")

# Load in the coco annotations
import json
import pandas as pd

import sys
import tempfile

import pickle
from PIL import Image
import numpy as np

import supervision as sv
import os

print("script message: loaded all libraries")

####################################################### Define Input and Output Paths ##########################################################
print("script message: getting import and output paths")

# input paths
predicting_data = sys.argv[1]
sam_results_by_image_path = sys.argv[2]
path_to_annotations = sys.argv[3]
sum_image_areas = sys.argv[4]
img_paths = sys.argv[5:]

# Output paths
temp_dir = tempfile.mkdtemp()

############################################################ Change Predictions  ###############################################################

print("script message: changing the predictions")
# changing the original labels in the annotations json file to the new labels from the predicting data

# Load in the predicting data df with the new predictions
predicting_names_labels_df = pd.read_csv(f"{predicting_data}/predictions_names_labels.csv")

# use the labels in the predicting data to replace the labels in the annotations json file

# read the patch_name 
# for each patch_name in the predicting_names_labels_df, get the image id and annotation id.
# These are the two numbers at the end of the patch_name where _#_#.png
# use the image id and annotation id to place the label from predicting_names_labels_df into data

# get the image id and annotation id
predicting_names_labels_df['image_id'] = predicting_names_labels_df['patch_name'].apply(lambda x: x.split("_")[-2])
predicting_names_labels_df['annotation_id'] = predicting_names_labels_df['patch_name'].apply(lambda x: x.split("_")[-1].split(".")[0])
class_ids = []
img_rgbs = []
annotated_image_paths = []
sum_aggregation_area = 0
sum_ice_area = 0

with open(path_to_annotations) as f:
    data = json.load(f)
    
#load in the pickle file of sam_results_by_image
with open(sam_results_by_image_path, 'rb') as f:
    sam_results_by_image = pickle.load(f)
    
# iterate over the predicting_names_labels_df
for index, row in predicting_names_labels_df.iterrows():
    # get the image id and annotation id
    image_id = int(row['image_id'])
    annotation_id = int(row['annotation_id'])
    # get the label
    label = int(row['label'])
    # find the annotation in data
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id and annotation['id'] == annotation_id:
            annotation['category_id'] = label
            class_ids.append(label)
            # get the area of the annotation if the category_id id=4 or 5
            if label == 4:
                sum_aggregation_area += annotation['area']
            elif label == 5:
                sum_ice_area += annotation['area']
            else:
                pass  
            break
        
# Save the data to a new json file
with open(f"{temp_dir}/updated_predictions_test.json", 'w') as f:
    json.dump(data, f)
    
############################################################ Show New Predictions  ###############################################################

# convert sum_image_areas to an integer
sum_image_areas = int(sum_image_areas)

aggregation_area_percent = round(sum_aggregation_area / sum_image_areas * 100, 2)
ice_area_percent = round(sum_ice_area / sum_image_areas * 100, 2)

print(f"aggregation_area: {aggregation_area_percent} %")
print(f"ice_area: {ice_area_percent} %")

for i,img_path in enumerate(img_paths):    
    img_rgb = Image.open(img_path)
    
    img_rgb = np.array(img_rgb).astype(np.uint8)
    img_rgbs.append(img_rgb)
    
mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.CLASS)

index_of_class_ids = 0

# print("script message: annotating the original images")
for i, img_result in enumerate(sam_results_by_image):
    # print(f"script message: annotating image {i}")
    # get the capsid_ids for the length of each sam_results_by_image[i]
    class_ids_perimage = class_ids[index_of_class_ids:index_of_class_ids+len(img_result)]
    
    detections = sv.Detections.from_sam(sam_result=img_result)
    detections.class_id = class_ids_perimage 

    annotated_img = mask_annotator.annotate(img_rgbs[i].copy(), detections=detections)
    
    output_path = os.path.join(temp_dir, f"annotated_{i}.png")
    Image.fromarray(annotated_img).save(output_path)
    annotated_image_paths.append(output_path)
    
    # last index of class_ids
    index_of_class_ids += len(img_result)
    print("script message: index of class ids", index_of_class_ids)
    
print("\n".join(annotated_image_paths))
print(f"{temp_dir}/updated_predictions_test.json")
