# syntax=docker/dockerfile:1
FROM python:3.10.8-bullseye
#Install linux

RUN apt-get install -y python3-pip
#Install PYTHON

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN patch -b /usr/local/lib/python2.7/dist-packages/tensorflow_estimator/python/estimator/keras.py estimator.patch

WORKDIR /ctrl
#DEFINES DIRECTORY FOR JUPYTER NOTEBOOKS FOR FILES

#EXPOSE 8888
#DEFAULT PORT FOR JUPYTER LAB SO IT CAN RUN 

#EXPOSE 8050

#CMD ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
#BUILDS JUPYTER LAB for development environment - run this if you want to develop the code

#CMD ["python", "/solmathdashboard/app/templates/app.py"]
#run this if you want to deploy the app upon launching the docker container