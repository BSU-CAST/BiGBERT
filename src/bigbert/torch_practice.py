import numpy as np
import torch
from torch import nn
from pathlib import Path
from tqdm import tqdm


def get_embeddings_dict(embedder="edu2vec"):
    if embedder == "edu2vec":
        embeddings = Path(__file__).resolve().parent.joinpath("data", "edu2Vec.txt")

    else:
        return

    embeddings_dict = {}
    with open(embeddings, 'r', encoding="utf8") as f:
        for line in tqdm(f, desc="Generating embeddings dict"):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    return embeddings_dict


if __name__ == "__main__":
    # Set up the Embedding layer
    embeddings_dict = get_embeddings_dict()
    embeddings_keys = list(embeddings_dict.keys())
    embeddings_tensor = torch.FloatTensor(embeddings_dict.values())
    edu2vec = nn.Embedding.from_pretrained(embeddings_tensor)

    print(embeddings_dict[embeddings_keys[1]])
    print(edu2vec(torch.LongTensor([1])))

    # embedding = nn.Embedding(1494958, 300)
    # edu2vec = embedding(embeddings)

    # Set up the 1D convolutional layer
    # cnn = nn.Conv1d()

    # Set up the BiGRU
    bigru = nn.GRU(10, 20, 2, bidirectional=True)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    output, hn = bigru(input, h0)
    print(output.shape)

    self_attn = nn.MultiheadAttention(20, 4)
    attn_output, attn_output_weights = self_attn(output, output, output)
