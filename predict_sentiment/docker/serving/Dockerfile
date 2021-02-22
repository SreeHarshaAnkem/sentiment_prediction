FROM tensorflow/serving
RUN mkdir -p /models
COPY models /models/
ENTRYPOINT ["tensorflow_model_server", "--port=8500",  "--rest_api_port=8501",  \
            "--model_name=sentiment",  "--model_base_path=/models/sentiment"]  