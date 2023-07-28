import os
import sys
import argparse
import shutil

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log
from imagenet import process_image

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("--input", type=str, default="/data/input", nargs='?', help="path to the input images folder")
parser.add_argument("--output", type=str, default="/home/nvidia/lung-cancer-recognition/data/output/latest_output", nargs='?', help="path to the output folder")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--topK", type=int, default=1, help="show the topK number of class predictions (default: 1)")
parser.add_argument("--folder", type=str, default="data/input", help="The input folder that the script will be ran in")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

total_answers = 0

correct_label = args.folder
print("="*20)
folder_path = os.path.join(os.getcwd(), args.folder)

data_list = os.listdir(folder_path)

if os.path.exists(args.output):
      
    # Delete Folder code
    shutil.rmtree(args.output)
    os.mkdir(args.output)

avg_confidence = 0
highest_confidence = 0
lowest_confidence = 101

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

    total_answers+=1
    avg_confidence += confidence

avg_confidence /= total_answers

print("average confidence: ", avg_confidence)
print("best confidence: ", highest_confidence)
print("worst confidence: ", lowest_confidence)