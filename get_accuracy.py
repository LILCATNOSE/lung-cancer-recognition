import os
import sys
import argparse
import shutil

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log
from imagenet import process_image

# get_accuracy.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/val/lung_scc/lungscc67.jpeg test.jpeg

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("--input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("--output", type=str, default="/home/nvidia/lung-cancer-recognition/data/output/lung_aca_output/", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--topK", type=int, default=1, help="show the topK number of class predictions (default: 1)")
parser.add_argument("--folder", type=str, default="lung_aca", help="The directory that the script will be ran in")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

correct_answers = 0
total_answers = 0

correct_label = args.folder
print("="*20)
folder_path = os.path.join(os.getcwd(), "data/val", args.folder)

data_list = os.listdir(folder_path)

if os.path.exists(args.output):
      
    # Delete Folder code
    shutil.rmtree(args.output)
    os.mkdir(args.output)

avg_confidence = 0
highest_confidence = 0
lowest_confidence = 0

for image in data_list:
# for i in range(0, len(os.listdir(os.path.join("data/val/", args.folder)))):
    image_path = os.path.join(folder_path, image)
    output_path_name = os.path.join(args.output, image)
    # ai_labels = process_image(data_list[i], args.output, args.network, args.topK)
    # print("get_accuracy", image)
    results = process_image(image_path, output_path_name, args.network, args.topK)

    ai_labels = results[0]
    confidence = results[1]

    # get highest and lowest confidence
    if confidence < lowest_confidence:
        lowest_confidence = confidence
    elif confidence > highest_confidence:
        highest_confidence = confidence

    avg_confidence += confidence

    # check if the ai guess is correct
    if correct_label.upper() in ai_labels:
        correct_answers += 1
        total_answers += 1
        print("Correct guess")
    else:
        total_answers += 1
        print("Incorrect guess")

accuracy = correct_answers/total_answers*100
avg_confidence /= total_answers

print("accuracy: ", accuracy)

print("average confidence: ", avg_confidence)
print("best confidence: ", highest_confidence)
print("worst confidence: ", lowest_confidence)