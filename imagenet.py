import sys
import argparse

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log


# Function that returns the confidence and predicted label
def process_image(inp, out, argtopK):
    
    # Load the recognition network (ImageNet)
    net = imageNet(model="model/resnet18.onnx", labels="model/labels.txt", 
                    input_blob="input_0", output_blob="output_0")
    
    # Define input and output vdeo sources and font
    input = videoSource(inp, argv=sys.argv)
    output = videoOutput(out, argv=sys.argv)
    font = cudaFont()

    # Capture the next image
    img = input.Capture()

    # Timeout
    if img is None:
        return  

    # Classify the image and get the topK predictions
    predictions = net.Classify(img, topK=argtopK)

    # Declare list of labels
    labels = []
    
    # Loop for every topK prediction
    for n, (classID, confidence) in enumerate(predictions):
        classLabel = net.GetClassLabel(classID)
        labels.append(classLabel)
        confidence *= 100.0

        print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")

        font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}", 
                        x=5, y=5 + n * (font.GetSize() + 5),
                        color=font.White, background=font.Gray40)
                            
    # Render the image
    output.Render(img)

    # Update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # Print out performance info
    net.PrintProfilerTimes()

    # Return confidence and labels
    return labels, confidence
