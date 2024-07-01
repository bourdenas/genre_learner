FROM tensorflow/tensorflow
ENV RUNNING_IN_DOCKER true

# Update Ubuntu
RUN apt clean && apt update && apt upgrade -y

# Install base packages and locales
RUN apt-get install --yes --no-install-recommends \
    ca-certificates

RUN apt-get remove python3-blinker -y

# Clean up cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Set the working directory
ARG APP_USER_HOME=/app
WORKDIR $APP_USER_HOME

COPY ./classifier ./classifier
COPY ./app.py ./app.py
COPY ./tags.keras ./tags.keras
COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ENV PORT 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:predict_genres(filename='tags.keras')"]
