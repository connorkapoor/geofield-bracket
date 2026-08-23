PY ?= .venv/Scripts/python.exe
DATA ?= data/v1

test:
	$(PY) -m pytest tests -m "not slow"

test-all:
	$(PY) -m pytest tests

data:
	$(PY) -m geofield.data.generate --n_brackets 50000 --n_csg 50000 \
		--physics_per_bracket 1 --mfg_dirs 2 --out $(DATA)

data-pilot:
	$(PY) -m geofield.data.generate --n_brackets 4000 --n_csg 8000 \
		--physics_per_bracket 1 --mfg_dirs 2 --out data/v0-pilot

data-check:
	$(PY) -m geofield.data.gallery --data $(DATA) --split train --n 64

train-a:
	$(PY) -m geofield.train.loop --config geofield/train/configs/stage_a.yaml

train-b:
	$(PY) -m geofield.train.loop --config geofield/train/configs/stage_b.yaml

train-baseline:
	$(PY) -m geofield.train.loop --config geofield/train/configs/baseline.yaml

train-flow:
	$(PY) -m geofield.train.flow_loop --config geofield/train/configs/stage_c.yaml

figures:
	$(PY) -m geofield.eval.figures --data $(DATA)

verify:
	$(PY) -m geofield.verify.report

demo:
	$(PY) -m geofield.demo.backend.app

.PHONY: test test-all data data-pilot data-check train-a train-b \
	train-baseline train-flow figures verify demo

