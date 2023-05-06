FROM ubuntu:kinetic
#FROM python:3.9

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
  && rm -rf /var/lib/apt/lists/*

COPY hw_06.py .
COPY . /app

CMD sleep 120 && sh /app/test.sh

#CMD ["sh", "/app/test.sh"]