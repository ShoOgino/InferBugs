FROM nvidia/cuda:10.1-base
RUN apt-get update
RUN apt-get upgrade
RUN apt-get install wget
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh