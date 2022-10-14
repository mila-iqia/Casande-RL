FROM python:3.7.6
WORKDIR /app

# permit http service

ENV PORT=80
EXPOSE 80

# permit installing private packages

ARG GEMFURY_TOKEN
ENV PIP_EXTRA_INDEX_URL https://pypi.fury.io/${GEMFURY_TOKEN}/dialogue/
ENV MLFLOW_TRACKING_URI="https://mlflow.dev.dialoguecorp.com"

# install poetry

ARG POETRY_VERSION="1.0.0"
RUN \
  pip install --upgrade pip \
  && pip install "poetry==${POETRY_VERSION}" \
  && poetry config virtualenvs.create false

# standard python project
COPY pyproject.toml poetry.lock ./
RUN poetry install -vvv --no-dev
COPY . .
ENTRYPOINT TBD
