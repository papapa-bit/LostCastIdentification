import numpy as np
embeds = np.load("found_embeddings.npy")
names = np.load("found_filenames.npy")
print(embeds.shape, names)