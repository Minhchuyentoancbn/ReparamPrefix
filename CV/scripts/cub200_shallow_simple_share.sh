export CUDA_VISIBLE_DEVICES=0

output_dir="output/cub200_shallow_simple_share"

for seed in "42" "44" "82" "100" "800"
do
    python train.py \
        --config-file configs/prompt/cub.yaml \
        DATA.BATCH_SIZE "64" \
        MODEL.TYPE "vit" \
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DROPOUT "0.0" \
        MODEL.PROMPT.KV_ONLY "True" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        SEED ${seed} \
        OUTPUT_DIR "${output_dir}/seed${seed}" \
        SOLVER.WEIGHT_DECAY "0.0001" \
        SOLVER.BASE_LR "1.0"
done