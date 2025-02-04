# Natural Language Processing Experiments

## Files:
    .
    ├── gpt2                          # Code for GPT2 style autoregressive LM
    │   ├── train_e2e.py              # high-level scripts to train.
    │   ├── train_control.py          # code that implements prefix-tuning.
    │   ├── trainer_prefix.py         # trainer code for the training loop. 
    │   ├── run_language_modeling.py  # training code (contains data loading, model loading, and calls trainer)
    │   ├── gen.py                    # high-level scripts to decode. 
    │   └── run_generation.py         # decoding code. 
    │
    ├── seq2seq                       # Code for encoder-decoder architecture
    │   ├── train_bart.py             # high-level scripts to train.
    │   ├── prefixTuning.py           # code that implements prefix-tuning.
    │   ├── finetune.py               # training code (contains data loading, model loading, and calls trainer)   
    │   ├── lightning_base.py         # helper code
    │   ├── utils.py                  # helper code
    │   └── callbacks.py              # helper code
    └── ...


To run the code for GPT2 style autoregressive LM, the code is in ``gpt2/``. This corresponds to the table-to-text experiments in the paper. 

To run the code for encoder-decoder architecture like BART,  the code is in ``seq2seq``. This corresponds to the summarization experiments in the paper. 

The two primary scripts I used to run my codes are `` gpt2/train_e2e.py`` (for table-to-text) and ``seq2seq/train_bart.py``(for summarization).
they are set to default of good hyperparameters, and can be used to tune hyperparameter :) 

-----------------------------------------------------
## Setup:

``cd transformer; pip install -e .``

Please extract `data.tar.gz` in the `data` directory.

-----------------------------------------------------
## Train via prefix-tuning:

- For table-to-text:

```python
cd gpt2;

# Deep share - E2E
python train_e2e.py --optim_prefix yes --preseqlen 5 --epoch 5 --learning_rate 0.0001 --mode data2text --bsz 10 --seed 101

# No share - E2E
python train_e2e.py --optim_prefix yes --preseqlen 5 --epoch 5 --learning_rate 0.01 --mode data2text --bsz 10 --seed 101 --init_shallow yes

# Deep share - WebNLG
python train_e2e.py --optim_prefix yes --preseqlen 5 --epoch 5 --learning_rate 0.0001 --mode webnlg --bsz 5 --seed 101

# No share - WebNLG
python train_e2e.py --optim_prefix yes --preseqlen 5 --epoch 5 --learning_rate 0.05 --mode webnlg --bsz 5 --seed 101 --init_shallow yes
```

- For summarization:
```python
cd seq2seq; 

# Deep share
python train_bart.py --mode xsum --preseqlen 200 --do_train yes --fp16 yes --bsz 16  --epoch 30  --gradient_accumulation_step 1 --learning_rate 0.00005  --mid_dim 512 --use_deep yes

# No share
python train_bart.py --mode xsum --preseqlen 200 --do_train yes --fp16 yes --bsz 16  --epoch 30  --gradient_accumulation_step 1 --learning_rate 0.0005  --mid_dim 512 --use_deep no
```