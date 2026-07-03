.PHONY: setup pipeline test

setup:
	pip install -r requirements.txt

pipeline:
	python main.py

test:
	pytest tests/
