FROM python:3.9

WORKDIR /app

ADD ./requirements.txt .
RUN pip3 install -r requirements.txt

ADD ./src/ ./src/
ADD ./tests/ ./tests/
ADD ./config.yaml ./
ADD ./data ./data
ADD kafka_2.13-3.4.0 ./
