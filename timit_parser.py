from pydub import AudioSegment
from pydub.playback import play
import os
import re



# reads the phonemes segmentation and splits the audio into corresponding chunks
def split_by_ph(phn_file, audio):
    for line in phn_file:
        ln = line.split()
        beg = int(ln[0])
        end = int(ln[1])
        phoneme = ln[2]

        chunk = ((audio[ (int(1000 * beg / frame_rate)) : int((1000 * end / frame_rate )) ]))
        add_audio_chunk(phoneme, chunk)


# exports the chunks into folders of its corresponding phonemes
# the resust directory contains subdirectories names after the different phonemes
# each subdirectory contains corresponding audio chunks
# info file is for storing number of chunks inside the directory
# you might need to change the paths and extensions for different corpuses
def add_audio_chunk(phn, chunk):
    if not os.access("result/" + phn, os.F_OK):
        os.mkdir("result/" + phn)
        f = open("result/" + phn + "/info", "w")
        i = 0
        f.write("%d" % i)
    else:
        f = open("result/" + phn + "/info", "r+")
        i = int(f.readline()) + 1
        f.seek(0, 0 )
        f.write('{}'.format(i))
    f.close()
    chunk.export("result/" + phn + '/' + str(i) + ".wav", format='wav')



frame_rate = 16000
directory = "TRAIN"

# iterates over each audio file in corpus. It's a horrendous piece of code
# for different corpuses you will probably need to change the paths and cycles
for subdir in os.listdir(directory):
    for subsubdir in os.listdir(directory + '/' + subdir):
        for elem in os.listdir(directory + '/' + subdir + '/' + subsubdir):
            if elem.endswith(".wav"):
                audio = AudioSegment.from_wav(directory + '/' + subdir + '/' + subsubdir + '/' + elem)
                phn_file = open(directory + '/' + subdir + '/' + subsubdir + '/' + elem[:-4] + ".PHN", "r")
                split_by_ph(phn_file, audio)

