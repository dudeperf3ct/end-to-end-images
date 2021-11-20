FROM nvcr.io/nvidia/pytorch:21.10-py3
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx
RUN pip3 install --upgrade pip
# cache requirements.txt
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
# copy everything else
COPY . /app
CMD bash
