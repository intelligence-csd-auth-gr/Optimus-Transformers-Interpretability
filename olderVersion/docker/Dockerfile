FROM ubuntu:18.04
FROM python:3.9.7
    
RUN apt-get update
RUN apt-get dist-upgrade -y

WORKDIR /usr/src/app

COPY data.zip .

RUN unzip data.zip

RUN pip install --upgrade setuptools

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

CMD ["voila", "VizTool.ipynb", "--port=8866", "--no-browser"]