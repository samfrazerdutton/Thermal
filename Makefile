.PHONY: install test test-unit test-integration lint doctor hardware clean

install:
	python -m pip install -e ".[dev]"

test: test-unit

test-unit:
	python -m pytest tests/unit -v

test-integration:
	python -m pytest tests/integration -v

doctor:
	python -m thermal.cli doctor

hardware:
	python -m thermal.cli hardware

clean:
	rm -rf build dist *.egg-info python/*.egg-info .pytest_cache
