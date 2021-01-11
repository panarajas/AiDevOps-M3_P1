FROM python:3.6.10

# set the working directory
RUN ["mkdir", "app"]
RUN ["mkdir", "export"]


WORKDIR "app"
RUN ["mkdir", "-p", "/opt/ml/input/data/training"]
RUN ["mkdir", "-p", "/tmp/pickle"]

# install code dependencies
COPY "requirements.txt" .
RUN ["pip", "install", "-r", "requirements.txt"]

COPY "BatchCreator_data_preprocessing.ipynb" .
#COPY BatchCreator.py /app/BatchCreator.py
COPY "Train.py" /app/Train.py

# install environment dependencies
ENV SHELL /bin/bash

# provision environment
#EXPOSE 8080

# VOLUME ["/home/tarun/Downloads" "/tmp"]
# If the following dependency is put in requirements, it gets ignored as 1.0.8 is already found but that version poses issues in code
RUN ["pip", "install", "--upgrade", "Keras-Applications==1.0.7"]
COPY BatchCreator.py /app/BatchCreator.py
COPY "Train.py" /app/Train.py

RUN apt-get update

RUN apt-get install -y \
    libgl1-mesa-glx

COPY "run.sh" .
RUN ["chmod", "+x", "./run.sh"]
COPY training-data/ /tmp/training-data/
RUN ["mkdir", "-p" , "/app/training-data"]
COPY training-data/* /app/training-data/

COPY "preparemodel.py" /app/preparemodel.py
RUN ["pip", "install", "boto3"]
ENTRYPOINT ["./run.sh"]
CMD ["train"]
