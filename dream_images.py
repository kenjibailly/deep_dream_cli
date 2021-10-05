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
layer 10: monkey's, lizards, snakes, duck

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

print("------------------------------------------------------")

# Folders into variables
deep_dream_images_ouput_folder = "deep_dream_images_output/"
deep_dream_images_input_folder = "deep_dream_images_input/"

print("------------------------------------------------------")
print("Choose a layer or experiment by choosing another layer.")
print("------------------------------------------------------")
print("layer 1: wavy")
print("layer 2: lines")
print("layer 3: boxes")
print("layer 4: circles?")
print("layer 5: eyes")
print("layer 6: dogs, bears, cute animals")
print("layer 7: faces, buildings")
print("layer 8: fish begin to appear, frogs/reptilian eyes.")
print("layer 10: Monkey's, lizards, snakes, duck")
print("------------------------------------------------------")

# Ask input from the user
input_layer = input("Choose a layer (default: random 1-10): ") 
iterations = input("How many iterations? (default: random 5-100): ")
recursive_level = input("How many recursive levels? (default: random 0-8): ")
print("------------------------------------------------------")

# If user doesn't choose (presses enter) the inputs will be random

if input_layer == "":
    input_layer = random.randint(1,10)
    layer_tensor = model.layer_tensors[input_layer]
else:
    layer_tensor = model.layer_tensors[int(input_layer)]    


if iterations == "":
    iterations = random.randint(5,100)
else:
    iterations = int(iterations)


if recursive_level == "":
    recursive_level = random.randint(0,8)
else:
    recursive_level = int(recursive_level)

# Store all the folders and files in folder "convert_frames" into the folders and files array respectively

files = [] 
folders = [] 
for (path, dirnames, filenames) in os.walk(deep_dream_images_input_folder):
    folders.extend(os.path.join(path, name) for name in dirnames)
    files.extend(os.path.join(path, name) for name in filenames)

frames_amount = len(files) # Amount of images stored in the folder
print('Files ({}):'.format(len(files)))

# Show the user the parameters being used

print("Starting Deep Dream with chosen parameters:")
print("Chosen parameters: ")
print('Layer = {}'.format(input_layer))
print('Iterations = {}'.format(iterations))
print('Recursive Level = {}'.format(recursive_level))
print("------------------------------------------------------")

for i in range(0, 9999999999999999):

    if i+1 > frames_amount:
        print(" ")
        print("------------------------------------------------------")
        print('All pictures have been saved, Deep Dream is done.')
        print('Check out {} '.format(deep_dream_images_ouput_folder))
        print("------------------------------------------------------")

        break

    else:
        print(i)
        print(files[i])
        file_name_and_folder = files[i]  # Load file name from files array
        
        file_name = file_name_and_folder.split('/')[1] # Split the file to get just the name of the file with the extension

        img_result = load_image(str(file_name_and_folder)) # Load the image

        img_result = recursive_optimize(layer_tensor=layer_tensor, image=img_result,
                        # how clear is the dream vs original image        
                        num_iterations = iterations, step_size=1.0, rescale_factor=0.5,
                        # How many "passes" over the data. More passes, the more granular the gradients will be.
                        num_repeats = recursive_level, blend=0.2)

        img_result = np.clip(img_result, 0.0, 255.0)
        img_result = img_result.astype(np.uint8)
        result = PIL.Image.fromarray(img_result, mode='RGB')

        result.save('{}{}.png'.format(deep_dream_images_ouput_folder, i)) # Save the image

        # Output to user
        print("")
        print('Picture {} saved.'.format(file_name))

        i += 1


        # result.show()