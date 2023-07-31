import os
import sys
import argparse
import shutil

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log
from imagenet import process_image

# Declare parser
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

# Setting the arguments for the parser
parser.add_argument("--input", type=str, default="data/input", nargs='?', help="path to the input images folder")
parser.add_argument("--output", type=str, default="/home/nvidia/lung-cancer-recognition/data/output/latest_output", nargs='?', help="path to the output folder")
parser.add_argument("--topK", type=int, default=1, help="show the topK number of class predictions (default: 1)")

# Parse the command line
try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# Get path to input folder
input_path = os.path.join(os.getcwd(), args.input)

data_list = os.listdir(input_path)

# Clear the output folder if it already contains images
if os.path.exists(args.output):
      
    shutil.rmtree(args.output)
    os.mkdir(args.output)

# Declare the confidence variables and total answers
avg_confidence = 0
lowest_confidence = 101
total_answers = 0

# Repeat for all images in input folder
for image in data_list:
    image_path = os.path.join(input_path, image)
    output_path = os.path.join(args.output, image)

    # Process the current image, get the results
    results = process_image(image_path, output_path, args.topK)

    # Parse the results
    ai_labels = results[0]
    confidence = results[1]

    # Get highest and lowest confidence
    if confidence < lowest_confidence:
        lowest_confidence = confidence

    # Count the amount of results
    total_answers += 1

    avg_confidence += confidence

# Calculate average confidence
avg_confidence /= total_answers

# Print the average and lowest confidence 
print("Average confidence: ", avg_confidence)
print("Worst confidence: ", lowest_confidence)