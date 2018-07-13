FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install keras tensorflow_hub

RUN  python -c 'import tensorflow_hub; tensorflow_hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/2")'

COPY . /app
WORKDIR /app

RUN ['python','run_service.py']
