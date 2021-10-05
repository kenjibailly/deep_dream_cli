# How to install Deep Dream on Windows 10

## 1. Download Python
To download Python, open Command Prompt and type in:
    
    python

## 2. Download Anaconda
Download here: https://www.anaconda.com/products/individual

## 3. Open Anaconda and run the following lines of code:

    conda create -n tf_gpu_env python=3.9
    conda activate tf_gpu_env
    conda install tensorflow-gpu -c anaconda
    conda install cudnn -c conda-forge 
    conda install cudatoolkit -c anaconda
    conda install -c conda-forge opencv ffmpeg

## 4. Check that Tensorflow is running

    conda activate tf-gpu

    python

    >>> import tensorflow as tf
    >>> print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    >>> exit()