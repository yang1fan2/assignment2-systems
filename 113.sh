# small
uv run python cs336_systems/benchmarking_script.py \
    --context_length 128 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --d_ff 3072 \
    --benchmark_options forward

uv run python cs336_systems/benchmarking_script.py \
    --context_length 128 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --d_ff 3072 \
    --benchmark_options backward

uv run python cs336_systems/benchmarking_script.py \
    --context_length 128 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --d_ff 3072 \
    --benchmark_options both

# medium
uv run python cs336_systems/benchmarking_script.py \
    --context_length 128 \
    --d_model 1024 \
    --num_layers 24 \
    --num_heads 16 \
    --d_ff 4096 \
    --benchmark_options forward

uv run python cs336_systems/benchmarking_script.py \
    --context_length 128 \
    --d_model 1024 \
    --num_layers 24 \
    --num_heads 16 \
    --d_ff 4096 \
    --benchmark_options backward

uv run python cs336_systems/benchmarking_script.py \
    --context_length 128 \
    --d_model 1024 \
    --num_layers 24 \
    --num_heads 16 \
    --d_ff 4096 \
    --benchmark_options both