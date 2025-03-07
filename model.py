import pandas as pd
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt

import keras as k
import keras.layers as l
from keras.regularizers import l2
from keras.utils.visualize_util import plot

########## PREPROCESSING ##########

# Read in a driving log
def read_datafile(folder, append_path=False):
    df = pd.read_csv(os.path.join(folder, 'driving_log_fixed.csv'))
    if append_path:
        df['left'] = folder + '/' + df['left']
        df['center'] = folder + '/' + df['center']
        df['right'] = folder + '/' + df['right']
    return df

# Shift input images and angles to augment additional steering data
# Based on https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ti3uedvq4
def shift_image(image, steer, trans_range, scale=1, y=False):
    # Randomly generate a translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2

    # Augment the steering angle
    steer_ang = steer + tr_x / trans_range * 2 * .2 * scale

    if y:
        # Also translate on the y axis
        tr_y = 40 * np.random.uniform() - 40 / 2
    else:
        tr_y = 0

    # Warp the image based on the translation
    trans_matrix = np.float32([[1, 0, tr_x],[0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, trans_matrix, (320, 160))

    return image_tr, steer_ang

# Generic preprocessing for all images to the net
# Also applies to test images from drive.py
def preprocess(img):
    # Remove top and bottom, and resize
    img = cv2.resize(img[50:-25, :, :], (200, 66))
    # Convert to HLS
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

def pi(img, title=False):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if title:
        plt.title(title)
    plt.show()

# Stream image data from disk and then apply augmentations
def load_and_augment(batch, trans=False):
    # How much the steering angle is offset with the left and right cameras
    # Tuned a lot, best value turned out to be 0.25
    angle_offset = 0.25

    # Load the three camera images for the selected batch
    left_images = [cv2.imread(f) for f in batch['left'].values]
    center_images = [cv2.imread(f) for f in batch['center'].values]
    right_images = [cv2.imread(f) for f in batch['right'].values]
    all_images = left_images + center_images + right_images

    pi(left_images[0], 'Left camera')
    pi(center_images[0], 'Center camera')
    pi(right_images[0], 'Right camera')

    # OpenCV fails silently if the file doesn't exist, this will check that all images were loaded correctly and if not raises an error.
    for f in all_images:
        if f is None:
            print(batch['left'].values[0])
            raise FileNotFoundError

    # Load angles from data file, and then apply the offset to the left and right images
    # Append these images together
    angles = batch['steering']
    all_angles = (angles + angle_offset).tolist() + angles.tolist() + (angles - angle_offset).tolist()

    # Flip all the images in the batch, and also reverse the steering angle for these images
    # This is good for keeping the angle distribution normal
    all_images.extend([np.fliplr(img) for img in all_images])
    all_angles.extend([-angle for angle in all_angles])

    pi(center_images[0], 'Original image')
    pi(np.fliplr(center_images[0]), 'Flipped image')

    # Apply translations to all the images - this is disabled for validation
    if trans:
        all_images_2 = []
        all_angles_2 = []
        for a, b in zip(all_images, all_angles):
            a, b = shift_image(a, b, 80, 1)
            all_images_2.append(a)
            all_angles_2.append(b)

        all_images = all_images_2
        all_angles = all_angles_2

    pi(center_images[0], 'Original image')
    pi(shift_image(center_images[0], 1, 80, 1)[0], 'Shifted image')

    # Apply preprocessing to all the images
    all_images = [preprocess(img) for img in all_images]

    # Bind to NumPy array
    all_images = np.array(all_images).astype(np.float32)
    all_angles = np.array(all_angles).astype(np.float32)

    return all_images, all_angles

# Generator which tields batches of training data, in order to prevent keeping everything in memory at once
def datagen(df, batch_size=32, aug=True):
    while True:
        # Sample a random set of data, and then load and augment the images for the model
        chunk = df.sample(n=batch_size).reset_index(drop=True)
        X, y = load_and_augment(chunk, trans=aug)

        # The augmentation has created thousands of images from our selected 32
        # Here, we select a random few to be in the actual batch
        idx = np.random.choice(X.shape[0], batch_size, replace=False)
        X = X[idx, :, :, :]
        y = y[idx]

        yield X, y

############ NEURAL NETWORK ##########

# A block containing a convolutional layer, an ELU activation and dropout
def add_convolutional_block(model, filters, size, subsample):
    model.add(l.Convolution2D(filters, size, size, subsample=(subsample, subsample),
            border_mode='valid', init='glorot_uniform',
            W_regularizer=l2(0.01)))
    model.add(l.ELU())
    model.add(l.Dropout(0.2))
    return model

# A block containing a dense layer, an ELU activation and dropout
def add_dense_block(model, size, dropout):
    model.add(l.Dense(size, init='normal'))
    model.add(l.ELU())
    if dropout:
        model.add(l.Dropout(dropout))
    return model

# Model based on NVIDIA's paper: End to End Deep Learning for Self-Driving cars
def nvidia_model(shape):

    model = k.models.Sequential()
    model.add(l.Lambda(lambda x: x / 127.5 - 1., input_shape=shape, output_shape=shape))

    add_convolutional_block(model, 24, 5, 2)
    add_convolutional_block(model, 36, 5, 2)
    add_convolutional_block(model, 48, 5, 2)
    add_convolutional_block(model, 64, 3, 1)
    add_convolutional_block(model, 64, 3, 1)

    model.add(l.Flatten())

    add_dense_block(model, 100, 0.2)
    add_dense_block(model, 50, 0.2)
    add_dense_block(model, 10, 0)

    model.add(l.Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model

########## TRAINING ##########

if __name__ == '__main__':
    # Initialise training and validation datasets
    df_train = read_datafile('additional_data', append_path=True)
    df_valid = read_datafile('udacity', append_path=True)

    print('Training samples {} Validation samples {}'.format(len(df_train), len(df_valid)))

    # Initialise a generator to show statistics - this is for user verification
    gen = datagen(df_train, 1024)
    X, y = next(gen)

    # Show three training images
    for i in range(3):
        plt.imshow(X[i])
        plt.show()

    # Show histograms of values
    plt.hist(X.flatten(), bins=50)
    plt.show()
    plt.hist(y, bins=50)
    plt.show()

    # Print what the MSE is when predicting the mean values
    # This is extremely useful to set a baseline score
    print(np.mean(np.square(y - y.mean())))
    gen = datagen(df_valid, 1024, aug=False)
    _, y = next(gen)
    print(np.mean(np.square(y - y.mean())))

    print('Compiling model ...')
    model = nvidia_model(_[0].shape)
    model.summary()
    plot(model, to_file='model.png', show_shapes=True)
    print('Training ...')

    gen_train = datagen(df_train, 32)
    gen_valid = datagen(df_valid, 32, aug=False)

    open('model.json', 'w').write(model.to_json())

    chkpnt = k.callbacks.ModelCheckpoint('checkpoints/weights2.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_weights_only=False)

    model.fit_generator(gen_train, samples_per_epoch=25000, nb_epoch=100, verbose=1, validation_data=gen_valid, nb_val_samples=2500, max_q_size=25, nb_worker=4, pickle_safe=True, callbacks=[chkpnt])
