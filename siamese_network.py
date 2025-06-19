# siamese_network.py
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50

# Load and freeze pre-trained ResNet50
def build_base_model():
    base_cnn = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    base_cnn.trainable = False
    return base_cnn

# Euclidean distance function
def euclidean_distance(vectors):
    a, b = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1, keepdims=True))

# Contrastive loss function (for training only)
def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    squared_pred = tf.square(y_pred)
    squared_margin = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * squared_pred + (1 - y_true) * squared_margin)

# Simaese model definition
def build_siamese_model():
    input_shape = (224, 224, 3)
    input_a = Input(shape=input_shape, name="input_a")
    input_b = Input(shape=input_shape, name='input_b')

    base_model = build_base_model()

    embedding_a = base_model(input_a)
    embedding_b = base_model(input_b)

    distance = layers.Lambda(euclidean_distance)([embedding_a, embedding_b])

    siamese_net = Model(inputs=[input_a, input_b], outputs=distance)
    siamese_net.compile(optimizer='adam', loss=contrastive_loss, metrics=["accuracy"])

    return siamese_net
