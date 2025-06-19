# train_siamese.py
import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.utils import img_to_array
from sklearn.model_selection import train_test_split
from siamese_network import build_siamese_model
from tensorflow.keras.applications.resnet50 import preprocess_input