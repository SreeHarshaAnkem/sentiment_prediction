build_serving:
	docker build -t sentiment_model -f docker/serving/Dockerfile .

build_data_prep:
	docker build -t sentiment_data_prep -f docker/data_preparation/Dockerfile .

build_nginx:
	docker build -t sentiment_nginx -f docker/nginx/Dockerfile .

compose_up:
	docker-compose -f docker/docker-compose.yaml up

run_serving:
	docker run -p 8501:8501 sentiment_model