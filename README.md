# Capsidize
***An app for automatic labeling of AAV capsid types in Cryo-TEM images***

This app can be used in different ways:

1) **No Coding:** Downloading the .dmg for macos like any other app
2) **Low Coding:** Cloning this repo to your device and running the GUI using electron 

## Option 1: Downloading the App
This file is quite large as it was made using electron, a software developing framework that allows us to use HTML, CSS, Javascript and Python together with a clean workflow. This is the easiest option for non-coders.

App Download Link: https://cmu.box.com/s/r8afk0ft8ae1eie99zg8ek0kin32gly1

Disclaimer: You may recieve apple security alerts. To use this app you will just need to ignore them and allow the app.

## Option 2: Run Cloned Repo 
To run the GUI from this source code you will need Node.js (Electron runs on Node) and Python 3.9. You can install Node.js from https://nodejs.org/en.

### Step 1: Clone The Repo

```
git clone https://github.com/lcgutier/Capsidize.git
```

### Step 2: Navigate To The Main Directory
```
cd Capsidize
```
### Step 3: Set Up The Environment

You can set up the environment in 4 different ways: with the requirements.txt, the environment.yml, the environment.txt, or with the below command line prompts

#### Option 1: Set up env with the requirements.txt
```
conda create --name capsidize1_env python=3.9 
conda activate capsidize1_env
pip install -r requirements.txt
```
#### Option 2: Set up env with the environment.yml
```
conda env create -f environment.yml
conda activate capsidize1_env
```
#### Option 3: Set up env with the below command line prompts
```
conda create --name capsidize1_env --file environment.txt
conda activate capsidize1_env
```
#### Option 4: Set up env with the below command line prompts
```
conda create --name capsidize1_env python=3.9 
conda activate capsidize1_env
conda install -c conda-forge tensorflow
conda install -c conda-forge opencv
conda install scikit-learn
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install torch
pip install torchvision
pip install matplotlib
pip install supervision
pip install opencv-rolling-ball
pip install numpy==1.24.4
pip install pandas
```
### Step 4: Download the Segment Anything Model Checkpoint
The Capsidize app uses the SAM to segment the objects in the images before classification. To use the source code for the Capsidize app you will need to download the ViT-H SAM model. This can be found on their github at the link below:

Information on the Segment Anything Model can be found here: https://github.com/facebookresearch/segment-anything

The checkpoint to the model can then be added to the engine folder

### Step 5: Install Node.js Dependencies in The GUI Folder

Make sure to navigate to the gui folder first using:
```
cd gui
```
Then install the needed dependencies
```
npm install
```
### Step 6: Run The App
place this line in the command prompt to create the GUI window.
``` 
npm start
```
From there you are free to upload images and detect

### Optional: If you would prefer to run the app with its binaries instead of the .py files you will need to add them separately

You will need to download the engine binaries or create them yourself using the .py files. 

Engine_Binaries Download Link: https://cmu.box.com/s/la3gt3ngoy6dhsj12tmi2swel5ox6eft

Once downloaded you can extract them into the engine folder



