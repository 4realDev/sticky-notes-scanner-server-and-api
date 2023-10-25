# Install TF2 object detection API
FROM python:3.11 as tf2-builder
WORKDIR /usr/local/src

RUN apt update && \
  apt install -y protobuf-compiler

RUN wget https://github.com/tensorflow/models/archive/refs/heads/master.zip && \
  unzip master.zip && \
  mkdir -p models/research && \
  cp -r models-master/research/object_detection models/research/object_detection

RUN cd models/research && \
  protoc object_detection/protos/*.proto --python_out=.

RUN cd models/research && \
  cp object_detection/packages/tf2/setup.py . && \
  pip install .

# Install our custom ML model
FROM python:3.11 as model-builder
WORKDIR /usr/local/src

RUN --mount=type=secret,id=gitlab-pat wget --header "Authorization: Bearer $(cat /run/secrets/gitlab-pat)" https://gitlab.cando-image.com/api/v4/projects/469/packages/generic/model/0.0.1/workspace.zip && \
  unzip workspace.zip

# Install python requirements
FROM python:3.11 as requirements-builder
WORKDIR /usr/local/src
COPY requirements.txt /usr/local/src
RUN cd /usr/local/src && ls -la  && pip install -r requirements.txt

# Add everything together
FROM python:3.11-slim
WORKDIR /usr/local/src

RUN apt update && \
  apt install libgl1 libglib2.0-0 -y && \
  apt clean

COPY --from=tf2-builder /usr/local/src/models /usr/local/src/models
COPY --from=tf2-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=model-builder /usr/local/src/workspace /usr/local/src/workspace
COPY --from=requirements-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY flaskr /usr/local/src/flaskr
COPY *.py /usr/local/src

# CMD python -m flask --app flaskr run --host=0.0.0.0 
CMD python -m waitress --call 'flaskr:create_app'
