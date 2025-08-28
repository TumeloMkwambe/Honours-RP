ENV_NAME = virenv
PYTHON_VERSION = 3.12

.PHONY: venv
venv:
	conda create -y -n $(ENV_NAME) python=$(PYTHON_VERSION)

.PHONY: install
install: venv
	conda run -n $(ENV_NAME) pip install --upgrade pip
	conda run -n $(ENV_NAME) pip install -r requirements.txt

.PHONY: clean
clean:
	conda remove -y -n $(ENV_NAME) --all
