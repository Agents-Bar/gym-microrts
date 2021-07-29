FROM ubuntu:20.04

RUN apt-get update && apt-get install -y python3 python3-pip openjdk-8-jdk-headless

# Install packages directly here to limit rebuilding on code changes.
# Packages are in order of likely rebuilds.
RUN pip3 install requests~=2.25.1 gym~=0.18.0
RUN pip3 install JPype1~=1.3.0
RUN pip3 install uvicorn~=0.14.0 fastapi~=0.67.0

COPY ./ /app

LABEL agents-bar-env-api=v0.1.0
LABEL gym-microrts=v0.1.0

CMD ["uvicorn", "gym_microrts.api:app", "--host", "0.0.0.0", "--port", "80", "--app-dir", "/app"]