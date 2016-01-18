FROM 4catalyzer/keras:tf

MAINTAINER grahama

RUN apt-get update && apt-get install -y build-essential git libatlas-base-dev libopencv-dev python3-skimage

ADD requirements.txt .

RUN pip install -r requirements.txt

RUN mkdir -p data/ src/

ADD src/ src/

# RUN git clone --recursive https://github.com/dmlc/mxnet
#
# RUN cd mxnet && make -j6
#
# RUN pip3 install -e python/