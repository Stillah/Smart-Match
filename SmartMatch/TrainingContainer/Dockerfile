FROM python:3.10

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN pip install --upgrade pip
COPY SmartMatch/TrainingContainer/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY SmartMatch/TrainingContainer /workspace/SmartMatch/TrainingContainer
COPY OCR /workspace/OCR

WORKDIR /workspace/SmartMatch/TrainingContainer

ENV IMG_FOLDER=/data/imgs
ENV OUTPUT_DIR=/data/output

CMD ["python", "HandwrittenTypedClassifier/train.py", "--img_folder", "/data/imgs", "--output_dir", "/data/output"]
