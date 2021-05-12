import torch


def _split(flattened_embedding, vectors_x_in, feats_x_in, vector_dim):
    embedding = torch.split(
        flattened_embedding, (feats_x_in, vectors_x_in * vector_dim), dim=-1
    )
    embedding = (embedding[0], embedding[1].reshape(embedding[1].shape[0], -1, vector_dim))
    return embedding

def _merge(embedding):
    flattened_embedding = torch.cat((embedding[0], embedding[1].reshape(embedding[1].shape[0], -1)), dim=-1)
    return flattened_embedding
