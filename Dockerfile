# app/Dockerfile

FROM python:3.10-slim-bookworm

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r /app/requirements.txt

RUN --mount=type=secret,id=SERPERDEV_API_KEY,mode=0444,required=true
RUN --mount=type=secret,id=OPENAI_KEY,mode=0444,required=true


# User
RUN useradd -m -u 1000 user
USER user
ENV HOME /home/user
ENV PATH $HOME/.local/bin:$PATH

WORKDIR $HOME
RUN mkdir app
WORKDIR $HOME/app
COPY . $HOME/app

EXPOSE 8501
CMD streamlit run app.py \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.fileWatcherType none
