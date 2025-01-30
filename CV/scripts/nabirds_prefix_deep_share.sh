export CUDA_VISIBLE_DEVICES=0

output_dir="output/nabirds_prefix_deep_share/"

for seed in "42" "44" "82" "100" "800"
do
    python train.py \
        --config-file configs/prompt/nabirds.yaml \
        DATA.BATCH_SIZE "64" \
        MODEL.TYPE "vit" \
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.PREFIX_TUNING "True" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        MODEL.PROMPT.SHARE_KV "True" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        SEED ${seed} \
        OUTPUT_DIR "${output_dir}/seed${seed}" \
        SOLVER.WEIGHT_DECAY "0.0001" \
        SOLVER.BASE_LR "10.0"
done