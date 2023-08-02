# Lung Cancer Classification Model

This model is used to classify two different types of lung cancer (Adenocarcenoma (ACA), Squamous cell carcinoma (SCC)) and healthy lung tissue (Normal (N)). It is trained on an ImageNet Resnet-18 model using transfer learning. This model will reduce the work needed to detect and effectively counter the two most common lung cancers as it only needs a microscopic sample of the lung tissue in to give an accurate prediction (average accuracy of 96.7%).


## Examples of output
<img src="https://i.imgur.com/SFtH1wZ.jpg" width="500" height="500">

##### All images used for training and testing including the above are taken from the following Kaggle dataset: https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

## The Model
The model was trained on a 2GB Jetson Nano with a pre-flashed SD card from the NVIDIA website and it uses ImageNet in order to classify between ACA, SCC, and N tissue. It was trained on 4200 augmented images of 750 cancerous tissue samples. The program can process as many images as you want at once. The input folder should contain these. The output images will contain the prediction as well as the % confidence. It is up to the user to interepret the results through the confidence score and guess. There will additionally be an output of average confidence alongside the lowest score which can be found in the terminal.

[Video explanation here](https://youtu.be/JJe4Bj_vtak)

## How to execute
1. Connect your Jetson Nano via SSH on Visual Studio Code
2. Download all the files from this repository into a 'folder' for easy access
3. Add images of choice into /'folder'/data/input
4. To start the program, type cd into the 'folder'. 
5. Since we are using a pre-flashed SD card, there sould be a docker container. Run ./docker/run.sh to start the docker container
6. Finally, start the program by typing 'python3 script.py'. Let the script run
7. The output will be found in /'folder'/data/output/latest_output

##### Note: The latest_output folder is emptied every time the script is ran, please make sure you save your images before running again.

## Arguments 
For more customizable usage, the following arguments can be used:
1. --input / overwrite path to the input folder
2. --output / overwrite path to the output folder
3. --topK / show the topK number of class predictions (default: 1)
