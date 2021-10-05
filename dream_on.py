'''
Some info on various layers, so you know what to expect
depending on which layer you choose:

layer 1: wavy
layer 2: lines
layer 3: boxes
layer 4: circles?
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
import cv2
import os
import random as random
import glob

# Folders into variables
dream_on_output_folder = "dream_on_images_output/"
dream_on_image_input_folder = "dream_on_image_input/"

print("------------------------------------------------------")
print('To start the "Dream On" previous "Dream On" images will be deleted in the "dream_on_output" folder.')
delete_previous_dream_on = input('Do you wish to continue? (y/n):  ')
if delete_previous_dream_on == 'n' or delete_previous_dream_on == 'N' or delete_previous_dream_on == 'no' or delete_previous_dream_on == 'No':
    exit()
# elif not delete_previous_dream_on == 'y' or not delete_previous_dream_on == 'Y' or not delete_previous_dream_on == 'yes' or not delete_previous_dream_on == 'Yes' or not delete_previous_dream_on == "":
#     # continue
#     exit()
elif delete_previous_dream_on == 'y' or delete_previous_dream_on == 'Y' or delete_previous_dream_on == 'yes' or delete_previous_dream_on == 'Yes' or delete_previous_dream_on == "":
    print("------------------------------------------------------")
    print('Starting clean up of previous "Dream On" session. ')

    clean_up_deep_dream_on_images = glob.glob('{}*'.format(dream_on_output_folder))
    for file in clean_up_deep_dream_on_images:
        os.remove(file)
        print(file)
print("------------------------------------------------------")
print('Clean up finished. Starting parameter choice.')
print("------------------------------------------------------")

print("Choose a layer or experiment by choosing another layer.")
print("------------------------------------------------------")
print("layer 1: wavy")
print("layer 2: lines")
print("layer 3: boxes")
print("layer 4: circles?")
print("layer 6: dogs, bears, cute animals")
print("layer 7: faces, buildings")
print("layer 8: fish begin to appear, frogs/reptilian eyes.")
print("layer 10: Monkey's, lizards, snakes, duck")
print("------------------------------------------------------")

# Ask input from the user
input_layer = input("Choose a layer (default: random 1-10): ") 
iterations = input("How many iterations? (default: random 5-100): ")
recursive_level = input("How many recursive levels? (default: random 1-8): ")

if recursive_level == '0':
    print('Recursive Level 0 cannot be used for "Dream On", Recursive Level set to minimum: 1')

amount_of_frames = input("How many frames do you want to generate? (default: 50): ")
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
    recursive_level = random.randint(1,8)
elif recursive_level == '0':
    recursive_level = 1
else:
    recursive_level = int(recursive_level)

if amount_of_frames == "":
    amount_of_frames = 50
else:
    amount_of_frames = int(amount_of_frames)


file = [] 
for (path, dirnames, filenames) in os.walk(dream_on_image_input_folder):
    file.extend(os.path.join(path, name) for name in filenames)
file = file[0]

img=PIL.Image.open(file)
w,h=img.size    # w=Width and h=Height
x_size = w
y_size = h

created_count = 0
max_count = amount_of_frames


print("Starting Deep Dream with chosen parameters:")
# Show the user the parameters being used
print('Layer = {}'.format(input_layer))
print('Iterations = {}'.format(iterations))
print('Recursive Level = {}'.format(recursive_level))
print('Amount of frames = {}'.format(amount_of_frames))
print("------------------------------------------------------")

for i in range(0, 9999999999999999):

    if os.path.isfile('{}/img_{}.png'.format(dream_on_output_folder, i+1)):
        print('{} already exists, continuing along...'.format(i+1))

    else:
        img_result = load_image(filename='{}'.format(file))

        # this impacts how quick the "zoom" is
        x_trim = 2
        y_trim = 1

        img_result = img_result[0+y_trim:y_size-y_trim, 0+x_trim:x_size-x_trim]
        img_result = cv2.resize(img_result, (x_size, y_size))

        # Use these to modify the general colors and brightness of results.
        # results tend to get dimmer or brighter over time, so you want to
        # manually adjust this over time.

        # +2 is slowly dimmer
        # +3 is slowly brighter
        img_result[:, :, 0] += 2  # reds
        img_result[:, :, 1] += 2  # greens
        img_result[:, :, 2] += 2  # blues

        img_result = np.clip(img_result, 0.0, 255.0)
        img_result = img_result.astype(np.uint8)

        img_result = recursive_optimize(layer_tensor=layer_tensor,
                                        image=img_result,
                                        num_iterations=iterations,
                                        step_size=1.0,
                                        rescale_factor=0.7,
                                        num_repeats=recursive_level,
                                        blend=0.2)

        img_result = np.clip(img_result, 0.0, 255.0)
        img_result = img_result.astype(np.uint8)
        result = PIL.Image.fromarray(img_result, mode='RGB')
        result.save('{}{}.png'.format(dream_on_output_folder, i+1))
        print('')
        print('Frame {}{}.png saved.'.format(dream_on_output_folder, i+1))

        created_count += 1
        if created_count > max_count -1:
            print('')
            print("------------------------------------------------------")
            print('Dream On finished. Check out {}'.format(dream_on_output_folder))
            print("------------------------------------------------------")
            break