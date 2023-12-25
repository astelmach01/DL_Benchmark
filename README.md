# DL_Benchmark

A benchmark on training deep learning models.

This trains a Vision Transformer, [ViT_B_16](https://pytorch.org/vision/main/models/vision_transformer.html), on CIFAR-100 for 1 epoch, 64 batch size.

## Setup

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## To Run

```shell
python3 benchmark.py
```

## Stats

| Device | Processor | RAM | Storage | Batch Size | Time (h:m:s) |
| --- | --- | --- | --- | --- | --- |
| MPS | M2 Pro | 32GB | 1TB SSD | 64 | 0:33:33 |
