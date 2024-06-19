import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Layer
from tensorflow.keras.models import Model

#################################################################################################################################################################
data_dir = '/home/yusuf/Desktop/Drill_Failure_Detection/Data/mtf2000'
img_size = (224, 224)

images = []
labels = []

# Load images and labels
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".jpeg"):
            image = Image.open(os.path.join(root, file))
            image = image.resize(img_size)
            image = np.array(image) / 255.0
            images.append(image)
            # Label: 1 for 'normal' (positive), 0 for 'anomaly' (negative)
            label = 1 if root.split('/')[-1] == 'normal' else 0
            labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15, random_state=42, stratify=labels)

# Further splitting train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Length of test set:", len(X_test))
print("Length of train set:", len(X_train))
print("Length of validation set:", len(X_val))


# Building the embedding layer
def make_embedding():
    inp = Input(shape=(224, 224, 3), name='input_image')

    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D((2, 2), padding='same')(c1)

    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D((2, 2), padding='same')(c2)

    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D((2, 2), padding='same')(c3)

    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=inp, outputs=d1, name='embedding')


# Building the Siamese Network
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.abs(input_embedding - validation_embedding)

def make_siamese_model():
    input_image = Input(shape=(224, 224, 3), name='input_image')
    validation_image = Input(shape=(224, 224, 3), name='validation_image')

    embedding = make_embedding()

    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

embedding = make_embedding()
model = make_siamese_model()

# Training the Siamese Neural Network
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

def train_step(input_image, validation_image, label):
    with tf.GradientTape() as tape:
        yhat = model([input_image, validation_image], training=True)
        loss = binary_cross_loss(label, yhat)

    grad = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grad, model.trainable_variables))
    return loss

@tf.function
def train_step(input_image_pair, label):
    input_image, validation_image = input_image_pair  # Unpack the input tuple into individual images

    with tf.GradientTape() as tape:
        distances = model([input_image, validation_image], training=True)  # Pass both input images to the model
        loss = binary_cross_loss(label, distances)

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train(X_train, y_train, X_val, y_val, EPOCHS):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    for epoch in range(1, EPOCHS + 1):
        print('\nEpoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(X_train))

        for input_image_pair, label in train_dataset:
            loss = train_step(input_image_pair, label)
            progbar.update(progbar.value + len(input_image_pair))

        if epoch % 10 == 0:
            # Evaluate on validation set here if needed
            checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 50
batch_size = 32  # Set your desired batch size
train(X_train, y_train, X_val, y_val, EPOCHS)
