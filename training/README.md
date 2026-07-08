# TheUnduster training pipeline

Offline pipeline: harvest real defects, synthesize training data, train the
detector, export ONNX, benchmark. Nothing here ships in the app except the
exported .onnx files.

## Setup

    mise install
    uv sync

## Pipeline (run from training/)

1. Harvest defects from blank film scans (see DATA.md for capture):

       uv run python -m unduster_training.harvest data/blank_scans data/library

2. Train both variants (clean images: any large, defect-free photo set;
   1000+ images recommended, mixed subjects):

       uv run python -m unduster_training.train --variant colour \
           --clean-dir data/clean --library-dir data/library \
           --steps 20000 --batch 8 --out data/ckpt/colour.pt
       uv run python -m unduster_training.train --variant bw \
           --clean-dir data/clean --library-dir data/library \
           --steps 20000 --batch 8 --out data/ckpt/bw.pt

3. Export to ONNX (fails loudly if torch/ORT disagree):

       uv run python -m unduster_training.export data/ckpt/colour.pt data/onnx/colour.onnx
       uv run python -m unduster_training.export data/ckpt/bw.pt data/onnx/bw.onnx

4. Benchmark against the labelled roll (see DATA.md for labelling):

       uv run python -m unduster_training.benchmark \
           --frames data/benchmark/frames --labels data/benchmark/labels \
           --onnx ours-colour=data/onnx/colour.onnx --onnx ours-bw=data/onnx/bw.onnx \
           --competitor retouch4me=data/benchmark/retouch4me \
           --out-dir reports

## Release gate

A model ships only if, on the labelled roll, it beats the classical baseline
and the competitor columns on precision AND per-type recall is no worse than
the previous shipped model. The benchmark report is the record.

## Tests

    uv run pytest
