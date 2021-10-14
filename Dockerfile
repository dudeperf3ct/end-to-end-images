FROM python:3.7-slim-buster

RUN apt-get update && apt-get install -y python3-dev build-essential

WORKDIR /app/src

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "app:app"]