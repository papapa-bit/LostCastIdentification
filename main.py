# main.py 
from feature_extractor import extract_features

img_path = 'dataset/bento-front.jpg'
vector = extract_features(img_path)

print("Feature vector length:", len(vector))
print("Feature preview:", vector[:10])