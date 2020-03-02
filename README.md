# LingoCorps-phoneme-recognition
for LingoCorps

## Requirements
### 1. Get language corpus with transcript
There is a collection of corpora available to EU students: https://www.clarin.eu/resource-families/spoken-corpora) \
[cs](https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-916)
### 2. Converting the transcript into phonemes
[phonemizer](https://github.com/bootphon/phonemizer)\
[from Wictionary](https://github.com/jojolebarjos/wiktionary-phoneme)
### 3. Do phoneme segmentation
[Persephone](https://persephone.readthedocs.io/en/latest/index.html) - might be very useful, as it claims the quantity needs of audio to phones mapping are low.
### 4. Parse the audio files
For this we will use [pydub](https://github.com/jiaaro/pydub).
If it throws an error processing .WAV files, saying something about NIST data in header, run this in the directory with your audio files:\
`find . -name '*.WAV' | parallel -P20 sox {} '{.}.wav'`\
the script _timit_parser.py_ walks through the [TIMIT corpus](http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3) and extracts chunks of audio files into folders of corresoonding phonemes. TIMIT corpus already has phoneme segmentation\
_timit_parser_results.zip_ contains result output of the script above.\
### 5. ML moder training
_train.py_ tries to do that. Uses _librosa_ and _keras_\
There are different modes. Each mode teaches the model based by different metrics, just change the variable `mode` in the code. With very small amount of data, on four phonemes and ten epochs, the accuracy is as follows:\
0: 28 % (don't use this mode, it basically feeds the model the raw audio, i. g the amplitude over time\
1: 68 %\
2: 50 %\
3: 68 %\


### Blind paths
_pyAudioAnalysis_ tool is not tht much customizable and doesn't acceps audio files as short as we need.
### Notes
Those people do something similar to what we do: https://easypronunciation.com/en/
