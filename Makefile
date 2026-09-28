.PHONY: test demo benchmark lint lock fmt

test:
	python -m pytest tests -q -W ignore::UserWarning -p no:cacheprovider

demo:
	python clusterforge/examples/run_demo.py

benchmark:
	python -m clusterforge.cli benchmark --out benchmark.json

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff format .

lock:
	python -m pip freeze > requirements.lock.txt
