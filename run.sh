#!/bin/bash
if [[ "$1" = train ]]
then
    echo 'Starting run.sh...'
    # the code expects a folder named pickle in the given directory
    mkdir -p /tmp/pickle
    mkdir -p /opt/ml/model
    # python BatchCreator.py
    echo "listing and counting train data"
    #ls /opt/ml/data/training |wc -l
    python BatchCreator.py /tmp/pickle /app/training-data
    # for debugging purposes
    ls /tmp/pickle
    python Train.py /tmp/pickle
    echo 'Content of current directory where model should be present...'
    ls /app
    echo 'Copying model files to /opt/ml/model...'
    # mkdir -p export/adas/1
    # cp /app/model.h5 /opt/ml/model/model.h5
    # cp /app/model.json /opt/ml/model/model.json
    python preparemodel.py
    echo 'Prepared model'
    cp -rv ./export /opt/ml/model/export
    ls /opt/ml/model/export
fi
