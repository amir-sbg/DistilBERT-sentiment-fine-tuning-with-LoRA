.PHONY: install train test clean

install:
	python -m pip install -e .

train:
	python -m fine_tuning_llms.pipeline

test:
	python -m pytest -q

clean:
	rm -rf data artifacts reports .pytest_cache src/llm_fine_tuning_pipeline.egg-info
