# Python version for the Docker image
FROM python:3.12-slim 

## Set the working directory in the container
WORKDIR /code

## Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

## Install the requirements.txt file
RUN pip install -r /code/requirements.txt

# Copy the content of the app directory to /code/app in the container
COPY ./app /code/app

# Set the environment variables for wandb
ENV WANDB_API_KEY=""
ENV MODEL_PATH=""

EXPOSE 8080

# Start the application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]