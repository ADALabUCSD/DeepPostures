FROM python:3.11.10
RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y git

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install torch==2.4.1
RUN pip install numpy
RUN pip install pandas
RUN pip install scipy
RUN pip install h5py
RUN pip install scikit-learn==1.5.2
RUN pip install tqdm==4.66.6

WORKDIR /

ARG CACHE_DATE=2024-10-22
RUN git clone https://github.com/ADALabUCSD/DeepPostures.git

USER root:root
