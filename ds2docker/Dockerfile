FROM grahama/deep:latest

MAINTAINER grahama

RUN apt-get update && apt-get install -y build-essential git libatlas-base-dev libopencv-dev python-skimage

ADD requirements.txt .

RUN pip3 install -r requirements.txt

RUN mkdir -p data/ src/

ADD src/ src/
