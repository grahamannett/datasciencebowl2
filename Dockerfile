FROM nvidia/cuda

MAINTAINER grahama

RUN apt-get update && apt-get install -y software-properties-common wget

RUN add-apt-repository ppa:fkrull/deadsnakes

RUN apt-get update && apt-get install -y python3.5 python3.5-dev libncurses5-dev libblas-dev liblapack-dev gfortran

RUN wget https://bootstrap.pypa.io/get-pip.py

RUN python3.5 get-pip.py

RUN pip3 install setuptools numpy scipy --upgrade

ADD requirements.txt .

RUN pip3 install -r requirements.txt

RUN cd ~ && echo "[global]\ndevice=gpu0\nfloatX=float32\n\n[cuda]\nroot=/usr/local/cuda" > .theanorc