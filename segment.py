import torch
import numpy as np

# segment len penalty function
def pen(segment_length):
    return 1 - segment_length

# Simple implementation of dynamic programming based phoneme segmentation method given in
#   Towards unsupervised phone and word segmentation using self-supervised vector-quantized neural networks
#   (https://arxiv.org/abs/2012.07551, INTERSPEECH 2021)
# Author: Yuan Tseng (https://github.com/roger-tseng)
def segment(reps, kmeans_model, pen, lambd=35):
    '''
    Inputs:
    reps        :   Representation sequence from self supervised model
    kmeans_model:   Pretrained scikit-learn MiniBatchKMeans model
    pen         :   penalty function penalizing segment length (longer segment, higher penalty)
    lambd       :   penalty weight (larger weight, longer segment)

    Outputs:
    boundaries  :   List of tokens at right boundaries of segments 
                    (assuming token sequence starts from 1 to Tth token)
    label_token :   List of token labels for segments

    e.g. :

    If  tokens = [34, 55, 62, 83, 42]
        boundaries = [3, 5]
        label_token = [55, 83]

    then segmentation is :
    | 34 55 62 | 83 42 |
    |    55    |   83  |

    '''
    
    # array of distances to closest cluster center, size: token sequence len * num of clusters
    distance_array = np.square( kmeans_model.transform(reps) )
    alphas = [[0, None]]

    # Perform dynamic-programming-based segmentation
    for t in range(1,reps.shape[0]+1):

        errors = []
        closest_centers = []
        
        for segment_length in range(1,t+1):

            # array len = num of clusters
            # ith element is sum of distance from the last segment_length tokens until Tth token to the ith cluster center
            distance_subarray = distance_array[t-segment_length:t].sum(axis=0)

            closest_center = distance_subarray.argmin()
            error = alphas[t-segment_length][0] + distance_subarray.min() + lambd * pen(segment_length)

            closest_centers.append(closest_center)
            errors.append(error)

        errors = np.array(errors)
        alpha, a_min, closest = errors.min(), t-1-errors.argmin(), closest_centers[errors.argmin()]
        alphas.append([alpha, a_min, closest])

    # Backtrack to find optimal boundary tokens and label
    boundaries = []
    label_tokens = []
    tk = len(alphas)-1
    while (tk!=0):
        boundaries.append(tk)
        label_tokens.append(alphas[tk][2])
        tk = alphas[tk][1]  
    boundaries.reverse()
    label_tokens.reverse()

    return boundaries, label_tokens


if __name__ == "__main__":

    import s3prl.hub as hub
    import soundfile as sf
    import joblib

    # Read input audio
    utterance = sf.read('/home/rogert/data/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac')[0]
    utterance = torch.from_numpy(utterance).to(torch.float)

    # Obtain HuBERT 6th layer representations of input
    model = getattr(hub, 'hubert')()
    model.eval()
    reps = model([utterance])['hidden_state_6'].squeeze()
    reps = reps.detach().numpy()

    # Perform k-means clustering with pretrained scikit-learn k-means model at https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit
    K = 100 # num of k-means clusters
    kmeans_model = joblib.load('./km100.bin')

    # Predict vector quantized tokens
    tokens = kmeans_model.predict(reps).tolist()
    print(f"tokens: {tokens}")
    print(f"len(tokens): {len(tokens)}")

    boundaries, label_tokens = segment(reps, kmeans_model, pen, 35)
    print(f"boundaries: {boundaries}")
    print(f"label_tokens: {label_tokens}")
    print(f"Num of segments = {len(label_tokens)}")

