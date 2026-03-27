TRAJECTORY_DIR="trajectories_test_full"
OUTPUT_DIR="activations_test_full"

# Extraction parameters
LAYERS="7,15,23"
STEPS="all"
PROMPT_SUFFIX_INDICES="-3:-1"
OUTPUT_INDICES="-16:-14"

for SIZE_DIR in "$TRAJECTORY_DIR"/size*; do

  CUDA_VISIBLE_DEVICES=0 uv run interp-cli gather_activations \
          --trajectory-paths "$SIZE_DIR/*.json" \
          --output-dir "$OUTPUT_DIR" \
          --layers "$LAYERS" \
          --steps "$STEPS" \
          --prompt-suffix-indices "$PROMPT_SUFFIX_INDICES" \
          --output-indices "$OUTPUT_INDICES"
done