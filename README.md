# Training the Built-In Object Detection Model in Amazon SageMaker and running it on AWS IoT Greengrass

This material strives to provide a comprehensive set of notebooks to showcase an ML Object detection workflow. The workflow is being leveraged as part of a larger foundational CV/ML framework. This repo was forked from Angela Wang's original work and cooresponding blog posts. Angela's work was heavily leveraged in place of previously written notebooks on a similar topic. The fork presents the merge of original material with Angela's more comprehensive material.

**Object detection** is the process of identifying and localizing objects in an image. A typical object detection solution takes in an image as input and provides a bounding box on the image where an object of interest is, along with identifying what object the box encapsulates.

Many scenarios of object detection happen in places with limited connectivity/bandwidth to internet. Therefore, running object detection at the IoT Edge is a often a solution in these use cases. 

This repo contains useful scripts and Juypter notebooks from collecting training data from a webcam to data labeling, to building an object detection model using built-in SSD model from Amazon SageMaker, and finally, deploying it to run and make inference the edge using AWS IoT Greengrass.
  
  
## Architecture 
![architecture-diagram](./imgs/architeture-diagram.png)

## Sections


## License Summary

This sample code is made available under the MIT-0 license. See the LICENSE file.
