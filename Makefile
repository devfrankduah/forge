# Forge -- LoRA fine-tuning from scratch. Self-contained; NumPy only.
PY ?= python3

.PHONY: demo compare web test
compare:
	$(PY) -m forge.compare_dora
demo:
	$(PY) -m forge.demo
web:
	$(PY) -m forge.web
test:
	$(PY) -m pytest tests/ -q || $(PY) tests/test_forge.py
