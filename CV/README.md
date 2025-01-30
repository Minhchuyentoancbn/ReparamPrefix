# Computer Vision Experiments


## Environment settings

See `env_setup.sh` if you have any issues with the environment setup. 


## Structure of the this repo (key files are marked with 👉):

- `src/configs`: handles config parameters for the experiments.
  
  * 👉 `src/config/config.py`: <u>main config setups for experiments and explanation for each of them. </u> 

- `src/data`: loading and setup input datasets. The `src/data/vtab_datasets` are borrowed from 

  [VTAB github repo](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data).


- `src/engine`: main training and eval actions here.

- `src/models`: handles backbone archs and heads for different fine-tuning protocols 

    * 👉`src/models/vit_prompt`: <u>a folder contains the same backbones in `vit_backbones` folder,</u> specified for VPT. This folder should contain the same file names as those in  `vit_backbones`

    * 👉 `src/models/vit_models.py`: <u>main model for transformer-based models</u> ❗️Note❗️: Current version only support ViT, Swin and ViT with mae, moco-v3

    * `src/models/build_model.py`: main action here to utilize the config and build the model to train / eval.

- `src/solver`: optimization, losses and learning rate schedules.  
- `src/utils`: helper functions for io, loggings, training, visualizations. 
- 👉`train.py`: call this one for training and eval a model with a specified transfer type.
- 👉`tune_fgvc.py`: call this one for tuning learning rate and weight decay for a model with a specified transfer type. We used this script for FGVC tasks.
- 👉`tune_vtab.py`: call this one for tuning vtab tasks: use 800/200 split to find the best lr and wd, and use the best lr/wd for the final runs
- `launch.py`: contains functions used to launch the job.


## Datasets preperation:

- Fine-Grained Visual Classification tasks (FGVC): The datasets can be downloaded following the official links. We split the training data if the public validation set is not available. The splitted dataset can be found in `local_datasets` folder.

  - [CUB200 2011](https://data.caltech.edu/records/65de6-vp158)

  - [NABirds](http://info.allaboutbirds.org/nabirds/)

  - [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/)

  - [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)

  - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

- [Visual Task Adaptation Benchmark](https://google-research.github.io/task_adaptation/) (VTAB): see `VTAB_SETUP.md` for detailed instructions and tips.


## Pre-trained model preperation

[Download](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) and place the pre-trained Transformer-based backbones to `MODEL.MODEL_ROOT`. Note that you also need to rename the downloaded ViT-B/16 ckpt from `ViT-B_16.npz` to `imagenet21k_ViT-B_16.npz`.


## Experiments

See `demo.ipynb` for how to use this repo. The folder `scripts` contains the scripts we used to run the experiments on FGVC.

### Examples

```bash
# Use the scripts that we provided
bash scripts/cub200_prefix_deep_share.sh


# For Deep-share
python train.py \
    --config-file configs/prompt/cub.yaml \
    DATA.BATCH_SIZE "64" \
    MODEL.TYPE "vit" \
    MODEL.PROMPT.NUM_TOKENS "20" \
    MODEL.PROMPT.PREFIX_TUNING "True" \
    MODEL.PROMPT.DEEP "True" \
    MODEL.PROMPT.DROPOUT "0.1" \
    MODEL.PROMPT.SHARE_KV "True" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    SEED 42 \
    OUTPUT_DIR "cub200/seed42" \
    SOLVER.WEIGHT_DECAY "0.0001" \
    SOLVER.BASE_LR "25.0"


# For No-share
python train.py \
    --config-file configs/prompt/cub.yaml \
    DATA.BATCH_SIZE "64" \
    MODEL.TYPE "vit" \
    MODEL.PROMPT.NUM_TOKENS "20" \
    MODEL.PROMPT.PREFIX_TUNING "True" \
    MODEL.PROMPT.DEEP "True" \
    MODEL.PROMPT.DROPOUT "0.1" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    SEED 42 \
    OUTPUT_DIR "cub200/seed42" \
    SOLVER.WEIGHT_DECAY "0.0001" \
    SOLVER.BASE_LR "50.0"


# For Simple-share
python train.py \
    --config-file configs/prompt/cub.yaml \
    DATA.BATCH_SIZE "64" \
    MODEL.TYPE "vit" \
    MODEL.PROMPT.NUM_TOKENS "10" \
    MODEL.PROMPT.DEEP "True" \
    MODEL.PROMPT.DROPOUT "0.1" \
    MODEL.PROMPT.KV_ONLY "True" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    SEED 42 \
    OUTPUT_DIR "cub200/seed42" \


# For VTAB
python tune_vtab.py \
    --train-type "prompt" \
    --config-file configs/prompt/structured/dsprites_location.yaml \
    MODEL.TYPE "vit" \
    DATA.BATCH_SIZE "128" \
    MODEL.PROMPT.NUM_TOKENS "20" \
    MODEL.PROMPT.PREFIX_TUNING "True" \
    MODEL.PROMPT.DEEP "True" \
    MODEL.PROMPT.DROPOUT "0.1" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    OUTPUT_DIR "output/tune_vtab_dsprites_location_prefix_no_share"   
```


## License

The majority of VPT is licensed under the CC-BY-NC 4.0 license (see [LICENSE](https://github.com/KMnP/vpt/blob/main/LICENSE) for details). Portions of the project are available under separate license terms: GitHub - [google-research/task_adaptation](https://github.com/google-research/task_adaptation) and [huggingface/transformers](https://github.com/huggingface/transformers) are licensed under the Apache 2.0 license; [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) are licensed under the MIT license; and [MoCo-v3](https://github.com/facebookresearch/moco-v3) and [MAE](https://github.com/facebookresearch/mae) are licensed under the Attribution-NonCommercial 4.0 International license.












