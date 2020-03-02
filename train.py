import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras import Input
from keras.engine import Model
from keras.utils import to_categorical
from keras.layers import Dense, TimeDistributed, Dropout, Bidirectional, GRU, BatchNormalization, Activation, LeakyReLU, \
    LSTM, Flatten, RepeatVector, Permute, Multiply, Conv2D, MaxPooling2D

# following code if gpu optimization, might also cause problems elsewhere
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
def get_session(gpu_fraction=0.8):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ktf.set_session(get_session())

# number of classes, in our case distinct phonemes
class_num = 60

# mode of metrics fed to the model
# 0 for waw (bad metrics)
# 1 for spectrogram
# 2 for mel spectogram
# 3 for mffc
mode = 1

# number of epochs
epochs = 10

train_X = []
train_spectrograms = []
train_mel_spectrograms = []
train_mfccs = []
train_y = []

test_X = []
test_spectrograms = []
test_mel_spectrograms = []
test_mfccs = []
test_y = []

# crops the audio if its longer than i, else it padds it with zeros
pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))
pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))

# the lib apparently accepts only numeral classes, so we map our phonemes to numbers
phoneme_n = 0
# path to our train data
DATA = 'data5'
os.chdir(DATA)
for d in os.listdir():
    os.chdir(d)
    phoneme = d
    for f in os.listdir():
        if f.endswith(".wav"):
            # loads the audio file. sr is sample rate, we are keeping the original, therefore
            # sr=None
            wav, sr = librosa.load(f, sr=None)
            if mode == 0:
                # cropping the audio file to 3000 ms
                padded_x = pad1d(wav, 3000)
                train_X.append(padded_x)
            if mode == 1:
                spectrogram = np.abs(librosa.stft(wav))
                padded_spectogram = pad2d(spectrogram, 32)
                train_spectrograms.append(padded_spectogram)

            if mode == 2:
                mel_spectrogram = librosa.feature.melspectrogram(wav)
                padded_mel_spectrogram = pad2d(mel_spectrogram, 32)
                train_mel_spectrograms.append(padded_mel_spectrogram)

            if mode == 3:
                mfcc = librosa.feature.mfcc(wav)
                padded_mfcc = pad2d(mfcc, 32)
                train_mfccs.append(padded_mfcc)

            train_y.append(phoneme_n)

    os.chdir('..')
    phoneme_n += 1
    print(phoneme)

phoneme_n = 0
# path to our test data
TEST = 'test5'
os.chdir('../' + TEST)
for d in os.listdir():
    os.chdir(d)
    phoneme = d
    for f in os.listdir():
        if f.endswith(".wav"):
            wav, sr = librosa.load(f, sr=None)
            if mode == 0:
                padded_x = pad1d(wav, 3000)
                test_X.append(padded_x)

            if mode == 1:
                spectrogram = np.abs(librosa.stft(wav))
                padded_spectogram = pad2d(spectrogram, 32)
                test_spectrograms.append(padded_spectogram)

            if mode == 2:
                mel_spectrogram = librosa.feature.melspectrogram(wav)
                padded_mel_spectrogram = pad2d(mel_spectrogram, 32)
                test_mel_spectrograms.append(padded_mel_spectrogram)

            if mode == 3:
                mfcc = librosa.feature.mfcc(wav)
                padded_mfcc = pad2d(mfcc, 32)
                test_mfccs.append(padded_mfcc)

            test_y.append(phoneme_n)

    os.chdir('..')
    phoneme_n += 1
    print(phoneme)

test_y = to_categorical(np.array(test_y))
train_y = to_categorical(np.array(train_y))
print('train_y:', train_y.shape)
print('test_y:', test_y.shape)

if mode == 0:
    test_X = np.vstack(test_X)
    train_X = np.vstack(train_X)
    print('train_X:', train_X.shape)
    print('test_X:', test_X.shape)

    ip = Input(shape=(train_X[0].shape))
    hidden = Dense(128, activation='relu')(ip)
    op = Dense(class_num, activation='softmax')(hidden)

if mode == 1:
    train_X_ex = np.expand_dims(np.array(train_spectrograms), -1)
    test_X_ex = np.expand_dims(np.array(test_spectrograms), -1)
    print('train X shape:', train_X_ex.shape)
    print('test X shape:', test_X_ex.shape)

    ip = Input(shape=train_X_ex[0].shape)
    m = Conv2D(32, kernel_size=(4, 4), activation='relu', padding='same')(ip)
    m = MaxPooling2D(pool_size=(4, 4))(m)
    m = Dropout(0.2)(m)
    m = Conv2D(64, kernel_size=(4, 4), activation='relu')(ip)
    m = MaxPooling2D(pool_size=(4, 4))(m)
    m = Dropout(0.2)(m)
    m = Flatten()(m)
    m = Dense(32, activation='relu')(m)
    op = Dense(class_num, activation='softmax')(m)

if mode == 2:
    train_X_ex = np.expand_dims(np.array(train_mel_spectrograms), -1)
    test_X_ex = np.expand_dims(np.array(test_mel_spectrograms), -1)
    print('train X shape:', train_X_ex.shape)
    print('test X shape:', test_X_ex.shape)

    ip = Input(shape=train_X_ex[0].shape)
    m = Conv2D(32, kernel_size=(4, 4), activation='relu', padding='same')(ip)
    m = MaxPooling2D(pool_size=(4, 4))(m)
    m = Dropout(0.2)(m)
    m = Conv2D(64, kernel_size=(4, 4), activation='relu')(ip)
    m = MaxPooling2D(pool_size=(4, 4))(m)
    m = Dropout(0.2)(m)
    m = Flatten()(m)
    m = Dense(32, activation='relu')(m)
    op = Dense(class_num, activation='softmax')(m)

if mode == 3:
    train_X_ex = np.expand_dims(np.array(train_mfccs), -1)
    test_X_ex = np.expand_dims(np.array(test_mfccs), -1)
    print('train X shape:', train_X_ex.shape)
    print('test X shape:', test_X_ex.shape)

    ip = Input(shape=train_X_ex[0].shape)
    m = Conv2D(64, kernel_size=(4, 4), activation='relu')(ip)
    m = MaxPooling2D(pool_size=(4, 4))(m)
# m = Conv2D(128, kernel_size=(2, 2), activation='relu')(ip)
# m = MaxPooling2D(pool_size=(2, 2))(m)
    m = Flatten()(m)
    m = Dense(32, activation='relu')(m)
    op = Dense(class_num, activation='softmax')(m)



model = Model(input=ip, output=op)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

history = model.fit(train_X_ex,
          train_y,
          epochs=epochs,
          batch_size=32,
          validation_data=(test_X_ex, test_y))




# shows a graph of accuracy
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


