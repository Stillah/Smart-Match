FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1

COPY SmartMatch/OCRContainer/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip "setuptools<82" wheel \
    && pip install --prefer-binary -r /tmp/requirements.txt \
    && pip install --no-deps opencv-contrib-python-headless==4.10.0.84

COPY SmartMatch/OCRContainer /workspace/SmartMatch/OCRContainer
COPY SmartMatch/TrainingContainer /workspace/SmartMatch/TrainingContainer
COPY OCR /workspace/OCR

WORKDIR /workspace/SmartMatch/OCRContainer/Backend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
