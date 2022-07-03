all: black lint

.PHONY: black
black:
	cd src/ && black .
	cd tests/ && black .

.PHONY: lint
lint:
	cd src/ && flake8 . --select=E7,E9,F63,F7,F82 --show-source --statistics
	cd tests/ && flake8 . --select=E7,E9,F63,F7,F82 --show-source --statistics