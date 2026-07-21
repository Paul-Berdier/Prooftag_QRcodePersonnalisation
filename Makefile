.PHONY: install test lint run notebook build deploy status benchmark benchmark-e004 benchmark-e005 pause-vllm resume-vllm

install:
	python -m pip install -e '.[dev]'

test:
	pytest --cov=prooftag_qr --cov-report=term-missing

lint:
	ruff check .

run:
	uvicorn prooftag_qr.api:app --reload --port 8080

notebook:
	python -m jupyter lab notebooks/01_srpg_step_by_step.ipynb

build:
	docker build -t prooftag-qr:dev .

deploy:
	bash scripts/create-database-secret.sh
	kubectl apply -k deploy/k8s

status:
	kubectl get pods,services,pvc -n qr-core

benchmark:
	bash scripts/benchmark.sh

benchmark-e004:
	bash scripts/e004-guided-benchmark.sh

benchmark-e005:
	bash scripts/e005-srpg-benchmark.sh

pause-vllm:
	bash scripts/gpu-workload.sh pause-vllm

resume-vllm:
	bash scripts/gpu-workload.sh resume-vllm
