#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def process_image(inp, out, argnetwork, argtopK):
    # load the recognition network

    # note: to hard-code the paths to load a model, the following API can be used:
    
    net = imageNet(model="model/resnet18.onnx", labels="model/labels.txt", 
                    input_blob="input_0", output_blob="output_0")
    print("imagenet: ", inp)
    # create video sources & outputs
    input = videoSource(inp, argv=sys.argv)
    output = videoOutput(out, argv=sys.argv)
    font = cudaFont()

    # process frames until EOS or the user exits

    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        return  

    # classify the image and get the topK predictions
    # if you only want the top class, you can simply run:
    #   class_id, confidence = net.Classify(img)
    predictions = net.Classify(img, topK=argtopK)

    print("prediction", predictions)
    # draw predicted class labels
    labels = []
    print(is_iterable(predictions))
    print(is_iterable(enumerate(predictions)))

    
    for n, (classID, confidence) in enumerate(predictions):
        classLabel = net.GetClassLabel(classID)
        labels.append(classLabel)
        confidence *= 100.0

        print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")

        font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}", 
                        x=5, y=5 + n * (font.GetSize() + 5),
                        color=font.White, background=font.Gray40)
                            
    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    return labels, confidence
    # exit on input/output EOS
    # if not input.IsStreaming() or not output.IsStreaming():
    #     return labels