# make pairs.py
import os
import random
import itertools
from glob import glob
from sklearn.model_selection import train_test_split

# Setting dataset path
DATASET_DIR = "dataset/"
EXTENSIONS = [".jpg", ".png", "jpeg"]

# Helper to check valid image files
def is_image(filename):
    return any(filename.lower().endswith(ext) for ext in EXTENSIONS)

def load_image_paths():
    cat_dirs = [os.path.join(DATASET_DIR, d) for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    image_dict = {}
    for cat_dir in cat_dirs:
        label = os.path.basename(cat_dir)
        images = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if is_image(f)]
        if len(images) >= 2: #Ensure enough for a positive pair
            image_dict[label] = images
        return image_dict
    
def generate_pairs(image_dict, max_negatives=3):
    positive_pairs = []
    negative_pairs = []

    # Generate positive pairs
    for label, images in image_dict.items():
        for pair in itertools.combinations(images, 2):
            positive_pairs.append((pair[0], pair[1], 1))

    # Generate negative pairs (controlledd number)
    labels = list(image_dict.keys())
    for i, label1 in enumerate(labels):
        for label2 in labels[i+1:]:
            img1 = random.choice(image_dict[label1])
            img2 = random.choice(image_dict[label2])
            negative_pairs.append((img1, img2, 0))
            if max_negatives > 1:
                for _ in range(max_negatives - 1):
                    img1 = random.choice(image_dict[label1])
                    img2 = random.choice(image_dict[label2])
                    negative_pairs.append((img1, img2, 0))
    return positive_pairs, negative_pairs

def save_pairs_csv(pairs, output_path):
    with open(output_path, "w") as f:
        f.write("img1, img2, label\n")
        for img1, img2, label in pairs:
            f.write(f"{img1}, {img2}, {label}\n")

if __name__ == "__main__":
    image_dict = load_image_paths()
    pos_pairs, neg_pairs = generate_pairs(image_dict)
    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)

    save_pairs_csv(all_pairs, "image_pairs.csv") 
    print(f"Generatedd {len(pos_pairs)} positive pairs and {len(neg_pairs)} negative_pairs.")
    print("Saved to image_pairs.csv")
