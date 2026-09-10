set -e
python -m pip install --no-build-isolation -e .
pip install flash-linear-attention matplotlib
python benchmarks/bench_fwd.py
