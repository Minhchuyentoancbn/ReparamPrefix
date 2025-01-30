export CUDA_VISIBLE_DEVICES=0

output_dir="output/cub200_prefix_simple_share/"

for seed in "42" "44" "82" "100" "800"
do
    python train.py \
        --config-file configs/prompt/cub.yaml \
        DATA.BATCH_SIZE "64" \
        MODEL.TYPE "vit" \
        MODEL.PROMPT.NUM_TOKENS "10" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        MODEL.PROMPT.KV_ONLY "True" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        SEED ${seed} \
        OUTPUT_DIR "${output_dir}/seed${seed}"
done