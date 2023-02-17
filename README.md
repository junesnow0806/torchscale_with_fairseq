# torchscale with fairseq

## Setup

setup for torchscale

```bash
# Install the repo as a package:
git clone https://github.com/msranlp/torchscale.git
cd torchscale
pip install -e .
pip install git+https://github.com/shumingma/fairseq.git@moe
pip install git+https://github.com/shumingma/infinibatch.git
pip install iopath
pip install --upgrade numpy
```

## Example: Language Modeling

### How to run

```bash
PATH_TO_DATA="YOUR-DATA-DIR/wikitext-103"
python torchscale_lm.py     ${PATH_TO_DATA}     --num-workers 2     --activation-fn gelu     --share-decoder-input-output-embed     --validate-interval-updates 1000     --save-interval-updates 1000     --no-epoch-checkpoints     --memory-efficient-fp16     --fp16-init-scale 4     --arch lm_base     --task language_modeling     --sample-break-mode none     --tokens-per-sample 128     --optimizer adam --adam-betas "(0.9, 0.98)"     --adam-eps 1e-08     --clip-norm 0.0     --lr 5e-4     --lr-scheduler polynomial_decay     --warmup-updates 750     --dropout 0.1     --attention-dropout 0.1     --weight-decay 0.01     --batch-size 4     --update-freq 1     --required-batch-size-multiple 1     --total-num-update 50000     --max-update 50000     --seed 1 --ddp-backend=c10d
```

## Example: Machine Translation

### How to run

```bash
PATH_TO_DATA="YOUR-DATA-DIR/iwslt14.tokenized.de-en"
python  torchscale_translation.py     ${PATH_TO_DATA}     --arch mt_base --share-decoder-input-output-embed     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000     --dropout 0.3 --weight-decay 0.0001     --max-tokens 4096 --fp16
```

## About Env

If error raised when running the above, install apex as the error tells. As for error about `cuda_profiler_api.h` not found when using cudatoolkit installed with conda, use

```python
from torch.utils.cpp_extension import CUDA_HOME
print(CUDA_HOME)
```

then check the cuda home dir, and add the `cuda_profiler_api.h` file under the `include` dir. You can find this file under the root cuda `include` dir(the cuda installed not with conda).

