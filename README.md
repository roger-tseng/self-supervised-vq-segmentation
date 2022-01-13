# Self Supervised VQ Phoneme Segmentation
A simple implementation of the dynamic-programming-based phoneme segmentation method given in [Towards unsupervised phone and word segmentation using self-supervised vector-quantized neural networks](https://arxiv.org/abs/2012.07551) (INTERSPEECH 2021)

<p align="center">
<img src="./assets/paper_aligned.png" alt="Schema showing effectiveness of DP-based segmentation method on VQ-VAE codes."
width="800px"></p>

## Dependencies
- The [S3PRL toolkit](https://github.com/s3prl/s3prl)
- Pretrained K-means model (see [FAIRSEQ GSLM](https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit))

## TODO:
- Add forced alignment and Praat visualization demo
