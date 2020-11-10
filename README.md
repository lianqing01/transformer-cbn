Code is copied from https://github.com/sIncerass/powernorm 
# Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.7

The fairseq library we use requires PyTorch version >= 1.2.0.
Please follow the instructions [here](https://github.com/pytorch/pytorch#installation).

After PyTorch is installed, you can install fairseq with:
```
conda env create --file env.yml
python setup.py build develop
```

# Reproduction
Prepare the traning data:

The scripts for training and testing is located at `trans-scripts` folder. Please refer to [this page](trans-scripts/data-preprocessing/README.md) to preprocess and get binarized data or use the data we provided in the next section. To reproduce the results for Table.1 by yourself:

```
# IWSLT14 De-En
## To train the model
./trans-scripts/train/train-iwslt14.sh encoder_norm_self_attn encoder_norm_ffn decoder_norm_self_attn decoder_norm_ffn
example:
$ CUDA_VISIBLE_DEVICES=0 ./trans-scripts/train/train-iwslt14.sh power power layer layer
$ CUDA_VISIBLE_DEVICES=0 ./trans-scripts/train/train-iwslt14.sh batch batch layer layer
$ CUDA_VISIBLE_DEVICES=0 ./trans-scripts/train/train-iwslt14.sh layer layer layer layer

For CBN (I only can train the CBN with only used it in the self-attentaion encoder part and other for layer norm.
Other way leads to the model collapse even I tune the gradient clip.):
$ CUDA_VISIBLE_DEVICES=0 ./trans-scripts/train/train-iwslt14_cbn.sh cbn layer layer layer
For tuning the hyper-parameter in cbn, you can check the script: trans-scripts/train/train-iwslt14_cbn.sh
lr: 0.00015
lr-cbn: 0.00015
weight-decay_cbn: 1.
cbn-loss-weight: 0.1
gradient-clip: 0.1
```


