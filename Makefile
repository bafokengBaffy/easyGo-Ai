install:
	pip install -r requirements.txt

lint:
	black src tests
	flake8 src tests

test:
	pytest

start:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000
