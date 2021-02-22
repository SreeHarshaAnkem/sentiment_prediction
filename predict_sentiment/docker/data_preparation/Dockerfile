FROM python:3.8
RUN mkdir -p /usr/app
RUN chmod +x /usr/app
WORKDIR /usr/app
COPY requirements.txt .
RUN ["python3", "-m", "pip", "install", "--upgrade", "pip"]
RUN ["python3", "-m", "pip", "install", "-r", "requirements.txt"]
COPY prepare_data .
COPY predict.py .
COPY wsgi.py .
COPY uwsgi.ini .
COPY config .
RUN chown 1001:0 /usr/app && chmod g+wrx /usr/app
USER 1001
ENTRYPOINT ["uwsgi", "--ini", "uwsgi.ini"]
