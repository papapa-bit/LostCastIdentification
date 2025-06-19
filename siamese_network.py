import tenserflow as tf
from tensorflow.keras import layers, Model, Input 
from tensorflow.keras.applications import ResNet50

def build_base_model():
    base_cnn = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    base_cnn.trainable = False
    return base_cnn

def build_siamese_model():
    input_shape = (244, 244, 3)

    input_a = Input(shape=input_shape, name="input_a")
    input_b = Input(shape=input_shape, name="input_b")

    base_model = build_base_model()

    embedding_a = base_model(input_a)
    embedding_b = base_model(input_b)

    # Distance L1 for now
    l1_distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([embedding_a, embedding_b])
    output = layers.Dense(1, activation='sigmoid'(l1_distance))

    siamese_net = Model(inputs=[input_a, input_b], outputs=output)
    siamese_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return siamese_net
