# LingoCorps-phoneme-recognition
for LingoCorps

## Requirements
### 1. Get language corpus with transcript
There is a collection of corpora available to EU students: (https://www.clarin.eu/resource-families/spoken-corpora) \
cz: (https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-916)
### 2. Converting the transcript into phonemes
### 3. Do phoneme segmentation
### 4. Parse the audio files
For this we will use [pydub](https://github.com/jiaaro/pydub).
If it throws and error rpocessing .WAV files, saying something about NIST data in header, run this in the directory with your audio files:\
`find . -name '*.WAV' | parallel -P20 sox {} '{.}.wav'`\
the script timit_parser.py walks through the [TIMIT corpus](http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3) and extracts chunks of audio files into folders of corresoonding phonemes
