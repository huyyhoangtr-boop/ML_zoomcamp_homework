FROM python:3.13.9-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY ["model.bin", "predict.py", "train.py",  "requirements.txt", "send_request.py",  "./"]
CMD ["python", "send_request.py", "predict.py"]
EXPOSE 9000
ENTRYPOINT ["gunicorn" ,"--bind=0.0.0.0:9000","predict:app"]