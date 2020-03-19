# CV/ML Toolkit for Self Guided Object Detection with Tensorflow, Sagemaker, and Sagemaker Groundtruth.

This material strives to provide a comprehensive set of notebooks to showcase a standard ML Object detection workflow. The workflow is being leveraged as part of a larger foundational CV/ML framework. This repo was forked from Angela Wang's original work but substantially modified to account for (among other things) a custom tensorflow image, model optimization through Hyperparameter tuning jobs, and model inference. The conclusion of the excercise produces model artifacts that can be consumed by your application. 

### CV/ML Toolkit Objectives:
1. Data collection/procurement by way of capturing static frames from video.
1. Curating a good data-set to prepare for labeling
1. Dataset labeling techniques with Amazon Sagemaker Groundtruth.
1. Launching a custom tensorflow container in your account
1. Training against that container with a base model of your choice.
1. Running Model tuning jobs to optimize your model across a given range of optimizations.
1. Testing the accuracy of your model on new images.
1. Export your model for use in your application.


**Object detection** is the process of identifying and localizing objects in an image. A typical object detection solution takes in an image as input and provides a bounding box on the image where an object of interest is, along with identifying what object the box encapsulates.

Many scenarios of object detection happen in places with limited connectivity/bandwidth to internet. Therefore, running object detection at the IoT Edge is a often a solution in these use cases. 
  
  
## Architecture 
![architecture-diagram](./imgs/architeture-diagram.png)

## Sections


## License Summary

This sample code is made available under the MIT-0 license. See the LICENSE file.
