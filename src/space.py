from annoy import AnnoyIndex
import numpy as np

def cosine_dist(pred, target):
    dot = np.dot(pred, target)
    norm_pred = np.linalg.norm(pred)
    norm_target = np.linalg.norm(target)
    return dot / (norm_pred * norm_target)

def build_space(emdim, embdict):
    space = AnnoyIndex(emdim, metric='angular')
    for i, vec in enumerate(embdict):
        space.add_item(i, vec)
    return space
