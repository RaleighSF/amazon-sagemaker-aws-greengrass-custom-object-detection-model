FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        zip \
        unzip \
        git \
        ca-certificates \
        curl \
        && rm -rf /var/lib/apt-get/lists/*

#Install protobuff/protoc
RUN curl -OL https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
RUN unzip protoc-3.6.1-linux-x86_64.zip -d protoc3
RUN mv protoc3/bin/* /usr/local/bin/
RUN mv protoc3/include/* /usr/local/include/
RUN rm protoc-3.6.1-linux-x86_64.zip

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /resources /opt/ml/code
WORKDIR /opt/ml/code

RUN pip install --upgrade pip
RUN pip install matplotlib

#RUN cd tensorflow-models/research/slim \
RUN pip install -e tensorflow-models/research/slim

#Build and install Tensorflow's Object Detection API
WORKDIR tensorflow-models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN python setup.py build
RUN python setup.py install

WORKDIR /opt/ml/code

#Update the python path to include the object detection API
ENV PYTHONPATH=${PYTHONPATH}:tensorflow-models/research:tensorflow-models/research/slim:tensorflow-models/research/object_detection

RUN echo $PYTHONPATH

RUN pip install -U scikit-image
RUN pip install -U scikit-learn
RUN pip install -U cython
RUN pip install -U scipy
RUN pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
RUN rm -rf /root/.cache