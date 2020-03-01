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



test_speaker = 'theo'
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

pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))
pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))

DATA = 'data1'
for d in os.listdir(DATA):
    for f in os.listdir(DATA + '/' + d):
        if f.endswith(".wav"):
            phoneme = d
            wav, sr = librosa.load(DATA + '/' + d + '/' + f)
            padded_x = pad1d(wav, 30000)
            spectrogram = np.abs(librosa.stft(wav))
            padded_spectogram = pad2d(spectrogram,40)

            mel_spectrogram = librosa.feature.melspectrogram(wav)
            padded_mel_spectrogram = pad2d(mel_spectrogram,40)

            mfcc = librosa.feature.mfcc(wav)
            padded_mfcc = pad2d(mfcc,40)


            train_X.append(padded_x)
            train_spectrograms.append(padded_spectogram)
            train_mel_spectrograms.append(padded_mel_spectrogram)
            train_mfccs.append(padded_mfcc)
            train_y.append(phoneme)

TEST = 'test1'
for d in os.listdir(TEST):
    for f in os.listdir(TEST + '/' + d):
        if f.endswith(".wav"):
            phoneme = d
            wav, sr = librosa.load(TEST + '/' + d + '/' + f)
            padded_x = pad1d(wav, 30000)
            spectrogram = np.abs(librosa.stft(wav))
            padded_spectogram = pad2d(spectrogram,40)

            mel_spectrogram = librosa.feature.melspectrogram(wav)
            padded_mel_spectrogram = pad2d(mel_spectrogram,40)

            mfcc = librosa.feature.mfcc(wav)
            padded_mfcc = pad2d(mfcc,40)

            test_X.append(padded_x)
            test_spectrograms.append(padded_spectogram)
            test_mel_spectrograms.append(padded_mel_spectrogram)
            test_mfccs.append(padded_mfcc)
            test_y.append(phoneme)

test_X = np.vstack(test_X)
test_spectrograms = np.array(test_spectrograms)
test_mel_spectrograms = np.array(test_mel_spectrograms)
test_mfccs = np.array(test_mfccs)
test_y = to_categorical(np.array(test_y))

train_X = np.vstack(train_X)
train_spectrograms = np.array(train_spectrograms)
train_mel_spectrograms = np.array(train_mel_spectrograms)
train_mfccs = np.array(train_mfccs)
train_y = to_categorical(np.array(train_y))

print('train_X:', train_X.shape)
print('train_spectrograms:', train_spectrograms.shape)
print('train_mel_spectrograms:', train_mel_spectrograms.shape)
print('train_mfccs:', train_mfccs.shape)
print('train_y:', train_y.shape)
print()
print('test_X:', test_X.shape)
print('test_spectrograms:', test_spectrograms.shape)
print('test_mel_spectrograms:', test_mel_spectrograms.shape)
print('test_mfccs:', test_mfccs.shape)
print('test_y:', test_y.shape)


ip = Input(shape=(train_X[0].shape))
hidden = Dense(128, activation='relu')(ip)
op = Dense(10, activation='softmax')(hidden)
model = Model(input=ip, output=op)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_X,
          train_y,
          epochs=10,
          batch_size=32,
          validation_data=(test_X, test_y))



plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


