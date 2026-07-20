.PHONY: install test lint run build deploy status pause-vllm resume-vllm

install:
	python -m pip install -e '.[dev]'

test:
	pytest --cov=prooftag_qr --cov-report=term-missing

lint:
	ruff check .

run:
	uvicorn prooftag_qr.api:app --reload --port 8080

build:
	docker build -t prooftag-qr:dev .

deploy:
	bash scripts/create-database-secret.sh
	kubectl apply -k deploy/k8s

status:
	kubectl get pods,services,pvc -n qr-core

pause-vllm:
	bash scripts/gpu-workload.sh pause-vllm

resume-vllm:
	bash scripts/gpu-workload.sh resume-vllm
