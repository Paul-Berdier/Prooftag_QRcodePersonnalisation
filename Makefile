.PHONY: install test lint run notebook notebook-search notebook-optimize build build-notebook deploy status benchmark benchmark-e004 benchmark-e005 notebook-start notebook-stop pause-vllm resume-vllm

install:
	python -m pip install -e '.[dev]'

test:
	pytest --cov=prooftag_qr --cov-report=term-missing

lint:
	ruff check .

run:
	uvicorn prooftag_qr.api:app --reload --port 8080

notebook:
	python -m jupyter lab notebooks/02_generate_live_on_gpu.ipynb

notebook-search:
	python -m jupyter lab notebooks/03_srpg_parameter_search.ipynb

notebook-optimize:
	python -m jupyter lab notebooks/04_e007_contextual_optimizer.ipynb

build:
	docker build -t prooftag-qr:dev .

build-notebook: build
	docker build -f Dockerfile.notebook -t prooftag-qr-notebook:dev .

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

notebook-start:
	bash scripts/notebook-server.sh start

notebook-stop:
	bash scripts/notebook-server.sh stop

pause-vllm:
	bash scripts/gpu-workload.sh pause-vllm

resume-vllm:
	bash scripts/gpu-workload.sh resume-vllm
