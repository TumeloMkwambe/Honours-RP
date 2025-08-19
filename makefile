VENV_DIR = venv

.PHONY: venv
venv:
	python3 -m venv $(VENV_DIR)

.PHONY: install
install: venv
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt