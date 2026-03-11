FROM python:3.9.0-slim
WORKDIR /app
ADD requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ADD . /app
CMD ["python", "app.py"]