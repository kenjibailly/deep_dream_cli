from backend.deepdreamer import model, load_image, recursive_optimize
import backend.dream_create_video as dream_create_video
import numpy as np
import PIL.Image
import os
import cv2
import glob
import json

def start_deep_dream(deep_dream_frames_folder, fps, layer_tensor, input_layer, iterations, recursive_level, last_index, total_frames):

    # Store all the folders and files in folder "convert_frames" into the folders and files array respectively
    # if not success:

    files = [] 
    folders = [] 
    for (path, dirnames, filenames) in os.walk('backend/frames_to_deep_dream'):
        folders.extend(os.path.join(path, name) for name in dirnames)
        files.extend(os.path.join(path, name) for name in filenames)

    # determine width and height of first image
    img=PIL.Image.open(files[0])
    w,h=img.size    # w=Width and h=Height
    frame_width = w
    frame_height = h

    frames_amount = total_frames # Amount of images stored in the folder

    print('Total frames: {}'.format(frames_amount))
    print("------------------------------------------------------")


    print("Starting Deep Dream with chosen parameters:")
    # Show the user the parameters being used
    print('Layer = {}'.format(input_layer))
    print('Iterations = {}'.format(iterations))
    print('Recursive Level = {}'.format(recursive_level))
    print("------------------------------------------------------")


    for i in range(0, 9999999999999999):
        if i+1 > (frames_amount-last_index):
            print(" ")
            print("------------------------------------------------------")
            print('All frames have been saved, Deep Dream is done.')
            print("------------------------------------------------------")
            print('Starting video creation')
            print("------------------------------------------------------")

            # Create the video from the frames generated (dream_video.py)
            dream_create_video.create_video(deep_dream_frames_folder, frames_amount, frame_width, frame_height, fps)
            
            print('Start cleaning up frames.')
            print("------------------------------------------------------")

            clean_up_deep_dream_frames = glob.glob('backend/deep_dream_frames/*')
            for file in clean_up_deep_dream_frames:
                os.remove(file)
            

            print('All done. Check out your video in the "video_output" folder.')
            print("------------------------------------------------------")
            
            break

        else:

            file_name_and_folder = files[i]  # Load file name from files array
            
            file_name = file_name_and_folder.split('\\')[1] # Split the file to get just the name of the file with the extension

            img_result = load_image(str(file_name_and_folder)) # Load the image

            img_result = recursive_optimize(layer_tensor=layer_tensor, image=img_result,
                            # how clear is the dream vs original image        
                            num_iterations = iterations, step_size=1.0, rescale_factor=0.5,
                            # How many "passes" over the data. More passes, the more granular the gradients will be.
                            num_repeats = recursive_level, blend=0.2)

            img_result = np.clip(img_result, 0.0, 255.0)
            img_result = img_result.astype(np.uint8)
            result = PIL.Image.fromarray(img_result, mode='RGB')

            # result.save(deep_dream_frames_folder+file_name) # Save the image
            result.save('{}{}.png'.format(deep_dream_frames_folder, i+last_index)) # Save the image

            # Output to user
            print("")
            print('Frame {} saved.'.format(file_name))

            i += 1


            # result.show()