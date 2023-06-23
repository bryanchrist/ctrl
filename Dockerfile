# syntax=docker/dockerfile:1
FROM python:3.8-bullseye
#Install linux

RUN apt-get -y update
#update apt

#RUN apt-get install -y python3-pip
#Install PYTHON

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

WORKDIR /ctrl
#DEFINES DIRECTORY FOR JUPYTER NOTEBOOKS FOR FILES

#RUN patch -b /usr/local/lib/python3.8/site-packages/tensorflow_estimator/python/estimator/keras_lib.py /ctrl/estimator.patch

EXPOSE 8888
#DEFAULT PORT FOR JUPYTER LAB SO IT CAN RUN 

#EXPOSE 8050

CMD ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
#BUILDS JUPYTER LAB for development environment - run this if you want to develop the code

#CMD ["python", "/solmathdashboard/app/templates/app.py"]
#run this if you want to deploy the app upon launching the docker container