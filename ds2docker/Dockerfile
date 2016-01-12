FROM grahama/deep:latest 

MAINTAINER grahama

ADD requirements.txt .

RUN pip3 install -r requirements.txt

RUN mkdir -p data/ src/

ADD src/ src/
