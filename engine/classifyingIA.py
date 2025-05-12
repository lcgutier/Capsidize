############################################################# Import Libraries #################################################################
print("script message: starting classifyingIA.py")

# Load in the coco annotations
import cv2
import numpy as np

import pandas as pd

# For basic image analysis
import os
import numpy as np

# load in the saved IA model
import joblib 

import sys
import tempfile

from scipy.stats import skew
from scipy.stats import kurtosis

import matplotlib.pyplot as plt

print("script message: loaded all libraries")

####################################################### Define Input and Output Paths ##########################################################

# in files
patches_image_directory = sys.argv[1]
model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ia_models/svm_model.pickle")

# out files
temp_dir = tempfile.mkdtemp()

################################# Load in the Predicting Data #################################

predicting_names_labels_df = pd.read_csv(f"{patches_image_directory}/names_labels_df.csv")

# Get the lists of file names and labels
predicting_names = predicting_names_labels_df['patch_name'].tolist()
predicting_labels = predicting_names_labels_df['label'].tolist()


predicting_images = [cv2.imread(f"{patches_image_directory}/{i}") for i in predicting_names]
# resize the images to 224x224
predicting_images = [cv2.resize(i, (224, 224)) for i in predicting_images]

################################# Load in the Prototypical Capsids #################################

# load in the average images
avg_filled = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ia_models/avg_capsids/avg_filled.png"))
avg_partial = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ia_models/avg_capsids/avg_partial.png"))
avg_empty = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ia_models/avg_capsids/avg_empty.png"))

################################# Convert Predicting Data to Feature Set #################################
print("script message: computing features")

### For Images With a Background 
# compute features on the patches and assesmble the features into a csv with the associated labels
# features: mean, sd, median, mean of inner patch, sd of inner patch, median of inner patch, min, max, range, similarity to avg_filled, similarity to avg_partial, similarity to avg_empty
# Do this for each of the list of images: full_images, partial_images, empty_images

left_center_coords = (predicting_images[0].shape[0] // 3) # assuming a square and that all images are the same size
right_center_coords = (2 * predicting_images[0].shape[0] // 3)

avg_full_inner_patch = avg_filled[left_center_coords:right_center_coords, left_center_coords:right_center_coords]
avg_partial_inner_patch = avg_partial[left_center_coords:right_center_coords, left_center_coords:right_center_coords]
avg_empty_inner_patch = avg_empty[left_center_coords:right_center_coords, left_center_coords:right_center_coords]

################################### Function to Compute Features #################################
def compute_features(image_list, labels):
    # initialize the lists
    mean = []
    sd = []
    median = []
    min_val = []
    max_val = []
    range_val = []
    skewness = []
    kurtosis_val = []
    mode = []
    mean_inner = []
    sd_inner = []
    median_inner = []
    min_inner = []
    max_inner = []
    range_inner = []
    skewness_inner = []
    kurtosis_inner = []
    mode_inner = []
    sim_to_avg_full_inner = []
    sim_to_avg_partial_inner = []
    sim_to_avg_empty_inner = []
    sim_to_avg_full = []
    sim_to_avg_partial = []
    sim_to_avg_empty = []
    
    # Iterate over the patches
    for patch in image_list:
        inner_patch = patch[left_center_coords:right_center_coords, left_center_coords:right_center_coords]
        mean.append(np.mean(patch))
        sd.append(np.std(patch))
        median.append(np.median(patch))
        min_val.append(np.min(patch))
        max_val.append(np.max(patch))
        range_val.append(np.max(patch) - np.min(patch))
        skewness.append(skew(patch.flatten()))
        kurtosis_val.append(kurtosis(patch.flatten()))
        mode.append(np.argmax(np.bincount(patch.flatten())))
        mean_inner.append(np.mean(inner_patch))
        sd_inner.append(np.std(inner_patch))
        median_inner.append(np.median(inner_patch))
        min_inner.append(np.min(inner_patch))
        max_inner.append(np.max(inner_patch))
        range_inner.append(np.max(inner_patch) - np.min(inner_patch))
        skewness_inner.append(skew(inner_patch.flatten()))
        kurtosis_inner.append(kurtosis(inner_patch.flatten()))
        mode_inner.append(np.argmax(np.bincount(inner_patch.flatten())))
        sim_to_avg_full_inner.append(np.linalg.norm(inner_patch - avg_full_inner_patch))
        sim_to_avg_partial_inner.append(np.linalg.norm(inner_patch - avg_partial_inner_patch))
        sim_to_avg_empty_inner.append(np.linalg.norm(inner_patch - avg_empty_inner_patch))
        sim_to_avg_full.append(np.linalg.norm(patch - avg_filled))
        sim_to_avg_partial.append(np.linalg.norm(patch - avg_partial))
        sim_to_avg_empty.append(np.linalg.norm(patch - avg_empty))
        
    # normalize each of the statistics
    mean = (mean - np.mean(mean)) / np.std(mean)
    sd = (sd - np.mean(sd)) / np.std(sd)
    median = (median - np.mean(median)) / np.std(median)
    min_val = min_val / np.max(min_val)
    max_val = max_val / np.max(max_val)
    range_val = (range_val - np.mean(range_val)) / np.std(range_val)
    skewness = (skewness - np.mean(skewness)) / np.std(skewness)
    kurtosis_val = (kurtosis_val - np.mean(kurtosis_val)) / np.std(kurtosis_val)
    mode = (mode - np.mean(mode)) / np.std(mode)
    mean_inner = (mean_inner - np.mean(mean_inner)) / np.std(mean_inner)
    sd_inner = (sd_inner - np.mean(sd_inner)) / np.std(sd_inner)
    median_inner = (median_inner - np.mean(median_inner)) / np.std(median_inner)
    min_inner = (min_inner - np.mean(min_inner)) / np.std(min_inner)
    max_inner = (max_inner - np.mean(max_inner)) / np.std(max_inner)
    range_inner = (range_inner - np.mean(range_inner)) / np.std(range_inner)
    skewness_inner = (skewness_inner - np.mean(skewness_inner)) / np.std(skewness_inner)
    kurtosis_inner = (kurtosis_inner - np.mean(kurtosis_inner)) / np.std(kurtosis_inner)
    mode_inner = (mode_inner - np.mean(mode_inner)) / np.std(mode_inner)
    sim_to_avg_full_inner = (sim_to_avg_full_inner - np.mean(sim_to_avg_full_inner)) / np.std(sim_to_avg_full_inner)
    sim_to_avg_partial_inner = (sim_to_avg_partial_inner - np.mean(sim_to_avg_partial_inner)) / np.std(sim_to_avg_partial_inner)
    sim_to_avg_empty_inner = (sim_to_avg_empty_inner - np.mean(sim_to_avg_empty_inner)) / np.std(sim_to_avg_empty_inner)
    sim_to_avg_full = (sim_to_avg_full - np.mean(sim_to_avg_full)) / np.std(sim_to_avg_full)
    sim_to_avg_partial = (sim_to_avg_partial - np.mean(sim_to_avg_partial)) / np.std(sim_to_avg_partial)
    sim_to_avg_empty = (sim_to_avg_empty - np.mean(sim_to_avg_empty)) / np.std(sim_to_avg_empty)
        
    # Create a dataframe of the statistics
    stats = pd.DataFrame({
        'mean': mean,
        'sd': sd,
        'median': median,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'skewness': skewness,
        'kurtosis': kurtosis_val,
        'mode': mode,
        'mean_inner': mean_inner,
        'sd_inner': sd_inner,
        'median_inner': median_inner,
        'min_inner': min_inner,
        'max_inner': max_inner,
        'range_inner': range_inner,
        'skewness_inner': skewness_inner,
        'kurtosis_inner': kurtosis_inner,
        'mode_inner': mode_inner,
        'sim_to_avg_full_inner': sim_to_avg_full_inner,
        'sim_to_avg_partial_inner': sim_to_avg_partial_inner,
        'sim_to_avg_empty_inner': sim_to_avg_empty_inner,
        'sim_to_avg_full': sim_to_avg_full,
        'sim_to_avg_partial': sim_to_avg_partial,
        'sim_to_avg_empty': sim_to_avg_empty
    })
    
    # add the labels to the dataframe
    stats['label'] = labels
    
    return stats

####################################### Compute Features #######################################

# Compute features for the training data
train_img_analysis_features_df = compute_features(predicting_images, predicting_labels)

######################################## Load in the Model ############################################
print("script message: loading model")

def refit_strategy(cv_results):
    """Here we will define a function to get the parameters of the best estimator
    First, we will filter by precision using a threshold, and then we will pick the best estimator of the remaining based on the recall
    """
    precision_threshold = 0.70
    
    cv_results_ = pd.DataFrame(cv_results)
    
    # Filter by precision
    high_precision_cv_results = cv_results_[cv_results_['mean_test_precision'] > precision_threshold]
    
    if not high_precision_cv_results.empty:
        best_recall_index = high_precision_cv_results['mean_test_recall'].idxmax()
        return best_recall_index
    else:
        # if the precision is too low, we will just pick the best estimator based on the recall
        best_recall_index = cv_results_['mean_test_recall'].idxmax()
        return best_recall_index

# load the model
model = joblib.load(model_save_path)

print("script message: model loaded")

################################################################ Make Predictions #############################################################

print("script message: predicting labels")
    
X_test = train_img_analysis_features_df.drop(columns=['label'])
y_test = predicting_labels

y_pred = model.predict(X_test)

################################################################ Calculate Stats #############################################################
output_image_path = []

# get count of each label
values, counts = np.unique(y_pred, return_counts=True)
print(f"full_capsids: {counts[0]}")
print(f"partial_capsids: {counts[1]}")
print(f"empty_capsids: {counts[2]}")
print(f"broken_capsids: {counts[5]}")
print(f"total_capsids: {counts[0] + counts[1] + counts[2]}")
print(f"viable_fraction: {round((counts[0] / (counts[0] + counts[1] + counts[2]))*100, 2)}")

print("script message: values", values)

# Create a bar graph of the counts of capsids
# include only the first 3 values and counts
plt.figure()
plt.bar(values[:3], counts[:3], color=['lightskyblue', 'darkorchid', 'green'])
plt.xticks(values[:3], ['Full', 'Partial', 'Empty'])
plt.xlabel("Capsid Type")
plt.ylabel("Count")
plt.title("Count of Capsids")
# plt.show()

# save the bar graph
capsid_count_plot = os.path.join(temp_dir, "capsid_count_plot.png")
plt.savefig(capsid_count_plot)

output_image_path.append(capsid_count_plot)
print("\n".join(output_image_path))
################################################################ Save Predictions #############################################################

print("script message: saving the predictions as predictions_names_labels.csv")
# Save the new predictions
predictions_df = pd.DataFrame(data={'patch_name': predicting_names, 'label': y_pred})

# Save the predictions in a csv to temp_dir
predictions_df.to_csv(f"{temp_dir}/predictions_names_labels.csv", index=False) 
print(temp_dir)