# How to install Deep Dream on Windows 10 (Nvidia Cuda supported GPU)

Note: If your graphics card isn't CUDA supported, things will not work.

## 1. Download Python
To download Python, open Command Prompt and type in
    
    python

## 2. Download Anaconda
Download here https://www.anaconda.com/products/individual

## 3. Open Anaconda and run the following lines of code

    conda create -n tf_gpu_env python=3.9
    conda activate tf_gpu_env
    conda install tensorflow-gpu -c anaconda
    conda install cudnn -c conda-forge 
    conda install cudatoolkit -c anaconda
    conda install -c conda-forge opencv ffmpeg
    conda install -c conda-forge matplotlib

## 4. Check if Tensorflow is running on your gpu

    python

     import tensorflow as tf
     print('Num GPUs Available', len(tf.config.list_physical_devices('GPU')))
     exit()


# How to use these scripts

## 1. Open Anaconda
## 2. Navigate to the deep_dream folder by using 
    cd pathtodeep_dream

## 3. Activate the gpu environment
    conda activate tf-gpu

## 4. Run one of the following scripts


## - Convert image(s) to Deep Dream
1. Put all images in the images_to_deep_dream folder.
2. All converted deep dream images will be generated in the deep_dream_images_input folder.
3. All converted images can be found in the deep_dream_images_output folder.

        python dream_images.py

## - Video (mp4) to deep dream video
1. Put video in the video_to_deep_dream folder.
2. Generated video can be found in the video_output folder.

Tip If you want to stop the process press ctrl+c twice. When the video hasn't finished generating yet and you start the script again, it will continue where it has left of. 

    python dream_video.py

## - Generate video frames from picture
1. Put a picture in the dream_on_image_input folder.
2. Generated pictures can be found in the dream_on_images_output folder.

Tip If you want to stop the process press ctrl+c twice.

    python dream_on.py

## - Generate video from images  frames
1. Generate a video from the images  frames created from
    - dream_images.py
    - dream_on.py
2. Video can be found in the video_output folder.
    
        python dream_images_to_video.py