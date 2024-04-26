FROM python:3.9

RUN apt-get update && apt-get install -y libhdf5-serial-dev

WORKDIR /app

COPY . /app
COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip3 install -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
