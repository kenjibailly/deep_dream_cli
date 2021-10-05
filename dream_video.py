'''
Some info on various layers, so you know what to expect
depending on which layer you choose:

layer 1: wavy
layer 2: lines
layer 3: boxes
layer 4: circles?
layer 5: eyes
layer 6: dogs, bears, cute animals.
layer 7: faces, buildings
layer 8: fish begin to appear, frogs/reptilian eyes.
layer 10: Monkies, lizards, snakes, duck

Choosing various parameters like num_iterations, rescale,
and num_repeats really varies on which layer you're doing.


We could probably come up with some sort of formula. The
deeper the layer is, the more iterations and
repeats you will want.

Layer 3: 20 iterations, 0.5 rescale, and 8 repeats is decent start
Layer 10: 40 iterations and 25 repeats is good. 
'''

from backend.deepdreamer import model, load_image, recursive_optimize
import numpy as np
import PIL.Image
import random as random
import os
import cv2
import glob
import backend.initiate_start as initiate_start
import backend.start_deep_dream as start_deep_dream
import json


# Folders into variables
deep_dream_frames_folder = "backend/deep_dream_frames/"

print("------------------------------------------------------")
print("Checking if video hasn't fully finished yet")
print("------------------------------------------------------")

deep_dream_files = [] 
for (path, dirnames, filenames) in os.walk('backend/deep_dream_frames'):
    deep_dream_files.extend(os.path.join(path, name) for name in filenames)
deep_dream_files = len(deep_dream_files)

frames_files = [] 
for (path, dirnames, filenames) in os.walk('backend/frames_to_deep_dream'):
    frames_files.extend(os.path.join(path, name) for name in filenames)
frames_to_del = frames_files

# Delete the frames up to the length
if deep_dream_files > 0:
    print("Last video hasn't been finished. Starting where left of...")
    print("------------------------------------------------------")
    for i in range(deep_dream_files):
        print("removing... "+frames_to_del[i])
        if os.path.exists('{}'.format(frames_to_del[i])):
            os.remove('{}'.format(frames_to_del[i]))
        i += 1
    
    # Check the last parameters used to enable them again
    with open('backend/last_parameters.json') as file:
        last_parameters = json.load(file)
    print("Last used parameters loaded.")

    # Loading variables
    fps = last_parameters["fps"]
    input_layer = last_parameters["input_layer"]
    iterations = last_parameters["iterations"]
    recursive_level = last_parameters["recursive_level"]
    total_frames = last_parameters["total_frames"]
    layer_tensor = model.layer_tensors[int(input_layer)]

    print("------------------------------------------------------")
    print('Video FPS: {}'.format(fps))

    last_index = deep_dream_files 

    start_deep_dream.start_deep_dream(deep_dream_frames_folder, fps, layer_tensor, input_layer, iterations, recursive_level, last_index, total_frames)
    
else:
    # Continue process
    initiate_start.initiate_start(deep_dream_frames_folder)