# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM ubuntu:latest

ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
#RUN apk add python3
#RUN apk add py3-pip
#RUN apt-get update && \
#    apt-get -y install python3 python3-pip
#    pip3 install virtualenv && \
#    virtualenv -p python3 venv && \
#    source venv/bin/activate && \
RUN apt-get update
RUN apt-get -y install python3 python3-pip
RUN apt install -y htop libgl1-mesa-glx libglib2.0-0
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD gunicorn app:app --workers 1 --threads 4