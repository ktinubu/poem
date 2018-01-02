FROM python:3

RUN apt-get update --fix-missing

# Install 
RUN apt-get install -y python python-dev python-setuptools
RUN apt-get install -y python-pip
RUN pip install --upgrade pip
RUN mkdir /code
WORKDIR /code
ADD requirements.txt /code/
RUN pip install -r requirements.txt
ADD . /code/
