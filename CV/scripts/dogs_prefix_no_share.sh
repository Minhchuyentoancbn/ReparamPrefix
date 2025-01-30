export CUDA_VISIBLE_DEVICES=0

output_dir="output/dogs_prefix_no_share/"

for seed in "42" "44" "82" "100" "800"
do
    python train.py \
        --config-file configs/prompt/dogs.yaml \
        DATA.BATCH_SIZE "64" \
        MODEL.TYPE "vit" \
        MODEL.PROMPT.NUM_TOKENS "200" \
        MODEL.PROMPT.PREFIX_TUNING "True" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        SEED ${seed} \
        OUTPUT_DIR "${output_dir}/seed${seed}" \
        SOLVER.WEIGHT_DECAY "0.0" \
        SOLVER.BASE_LR "50.0"
done