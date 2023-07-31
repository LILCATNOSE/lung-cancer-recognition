# Lung Cancer Recognition Model

This model is used to classify two different types of lung cancer (Adenocarcenoma (ACA), Squamous cell carcinoma (SCC)) and healthy lung tissue (Normal (N)). It is trained on an ImageNet Resnet-18 model using transfer learning. This model will reduce the work needed to detect and effectively counter the two most common lung cancers as it only needs a microscopic sample of the lung tissue in order to accurately predict with an average accuracy of 96.7%.


## Examples of output
<img src="https://i.imgur.com/SFtH1wZ.jpg" width="500" height="500">

##### All images used for training and testing including the above are taken from the following Kaggle dataset: https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images


## The Model
The model was trained on a 2GB Jetson Nano with a pre-flashed SD card from the NVIDIA website and it uses ImageNet in order to classify between ACA, SCC, and N tissue. The program will take an input path to a folder and an output folder. The input folder should contain one or more images to process. The output image will contain the prediction as well as the % confidence. It is up to the user to interepret the results through the confidence score and guess. There will additionally be an output of average confidence alongside the highest and lowest scores which can be found in the terminal.


## How to execute
1. Connect your Jetson Nano via SSH on Visual Studio Code
2. Download all the files from this repository into a 'folder' for easy access
3. Add images of choice into /'folder'/data/input
4. To start the program, type cd into the 'folder'. 
5. Since we are using a pre-flashed SD card, there sould be a docker container. Run ./docker/run.sh to start the docker container
6. Finally, start the program by typing 'python3 script.py'. The output will be found in /'folder'/data/output/latest_output