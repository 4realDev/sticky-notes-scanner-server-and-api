FROM python:3.11
WORKDIR /usr/local/src

RUN apt update && \
  apt install libgl1 -y && \
  apt clean

RUN apt update && \
  apt install -y protobuf-compiler && \
  apt clean && \
  wget https://github.com/tensorflow/models/archive/refs/heads/master.zip && \
  unzip master.zip && \
  rm -rf master.zip && \
  mkdir -p models/research && \
  cp -r models-master/research/object_detection models/research/object_detection && \
  rm -rf models-master && \
  cd models/research && \
  protoc object_detection/protos/*.proto --python_out=. && \
  cp object_detection/packages/tf2/setup.py . && \
  pip install .

COPY requirements.txt /usr/local/src
RUN cd /usr/local/src && ls -la  && pip install -r requirements.txt

COPY *.py /usr/local/src

CMD python app.py
