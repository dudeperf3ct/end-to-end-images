FROM nvidia/cuda:10.2-base-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    cmake \
    ca-certificates \
    build-essential \
    python3-dev \
    python3-pip
RUN pip3 install --upgrade pip
# cache requirements.txt
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
# copy everything else
COPY . /app
CMD bash