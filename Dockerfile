
FROM python:3.11-slim as base

RUN mkdir /app
COPY . /app/
WORKDIR /app

# dependencies
RUN pip install pip-tools

# COPY requirements.txt 
RUN pip install -r requirements.txt

