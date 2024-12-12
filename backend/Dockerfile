FROM python:3.12-slim-bullseye

# Install system dependencies, 使用 --upgrade 来安装最新版本的 poetry
RUN pip install --upgrade pip
# 设置poetry的版本为 1.8.0
ENV POETRY_VERSION 1.8.0

RUN pip install "poetry==$POETRY_VERSION"

# Clear poetry cache
# RUN poetry cache clear


# Set work directory
WORKDIR /app

# Copy only requirements to cache them in docker layer
COPY poetry.lock pyproject.toml /app/

# Remove poetry.lock file to force re-resolve
RUN rm poetry.lock

# Disable the creation of virtual environments
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install

# Copy the current directory contents into the container at /app
COPY ./ /app/