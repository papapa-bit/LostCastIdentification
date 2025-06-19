from feature_extractor import extract_embedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the database of embeddings
embeddings = np.load("found_embeddings.npy")
filenames = np.load("found_filenames.npy")

# Test image (simulate a lost cat image)
test_image = "found_cats/bento-front.jpg"
query_embedding = extract_embedding(test_image).reshape(1, -1)

# Compare all known embeddings
similarities = cosine_similarity(query_embedding, embeddings)

# Get best match
top_match_index = similarities[0].argmax()
top_similarity = similarities[0][top_match_index]
top_file = filenames[top_match_index]

print(f"\nTest Image: {test_image}")
print(f"Top Match Found: {top_file}")
print(f"Cosine Similarity Score: {top_similarity:.4f}")
