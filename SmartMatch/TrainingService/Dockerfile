FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1

COPY SmartMatch/TrainingService/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip "setuptools<82" wheel && pip install --prefer-binary -r /tmp/requirements.txt

COPY SmartMatch/TrainingService /workspace/SmartMatch/TrainingService
COPY SmartMatch/TrainingContainer /workspace/SmartMatch/TrainingContainer
COPY OCR /workspace/OCR

WORKDIR /workspace/SmartMatch/TrainingService/app

EXPOSE 8100

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
