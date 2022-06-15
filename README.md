# Simple Unsupervised Phoneme Segmentation
An unofficial implementation of the phoneme segmentation method given in
[Towards unsupervised phone and word segmentation using self-supervised vector-quantized neural networks](https://arxiv.org/abs/2012.07551) (INTERSPEECH 2021)

This method requires no additional training, and can be easily applied on various speech representations using the [S3PRL toolkit](https://github.com/s3prl/s3prl).<br/>
It works by simultaneously minimizing frame-wise distance to nearest k-means cluster center and the number of phoneme-like segments in an utterance with dynamic programming. (see paper above for details)

<p align="center">
<img src="./assets/paper_aligned.png" alt="Schema showing effectiveness of DP-based segmentation method on VQ-VAE codes."
width="800px"></p>

## Notice
Segmentation accuracy is heavily dependent on the parameter lambda, of which the optimal value varies greatly between choice of self-supervised representations. 
**Lambda is set to 35 in default.**<br/>
Values between 20~50 get about 60% F1 score for the 6th layer of HuBERT.

Also note that while the phoneme-level segments are each assigned to a cluster center, the same phoneme in different segments are often assigned to different cluster centers, which means that **this method is less suitable for phoneme discovery.**

## Visualization with Praat
Given an audio file and its text transcript, we can use forced alignment to obtain supervised word/phoneme boundaries, to visualize our method's accuracy.
Detailed steps are given in `demo.ipynb`.

<p align="center">
<img src="https://user-images.githubusercontent.com/67882177/163578210-a6240abd-64a5-48b7-9e9a-46565645a638.png" alt="Schema showing repreoduction results on HuBERT in Praat."
width="800px"></p>

## Dependencies
- The [S3PRL toolkit](https://github.com/s3prl/s3prl)
- Pretrained K-means model (see [FAIRSEQ GSLM](https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit))

## TODO:

- [x] Add forced alignment and Praat visualization demo
- [ ] Add F1 score calculation
