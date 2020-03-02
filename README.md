# LingoCorps-phoneme-recognition
for LingoCorps

## Requirements
### 1. Get language corpus with transcript
There is a collection of corpora available to EU students: https://www.clarin.eu/resource-families/spoken-corpora \
[cs](https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-916)
### 2. Converting the transcript into phonemes
[phonemizer](https://github.com/bootphon/phonemizer)\ - investigate the _espeak_ backend too, I think it's very useful.
[from Wictionary](https://github.com/jojolebarjos/wiktionary-phoneme)
### 3. Do phoneme segmentation
[Persephone](https://persephone.readthedocs.io/en/latest/index.html) - might be very useful, as it claims the initial quantity of phonemes-segmented audio is low.
### 4. Parse the audio files
For this we will use [pydub](https://github.com/jiaaro/pydub).
If it throws an error processing .WAV files, saying something about NIST data in header, run this in the directory with your audio files:\
`find . -name '*.WAV' | parallel -P20 sox {} '{.}.wav'`\
the script _timit_parser.py_ walks through the [TIMIT corpus](http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3) and extracts chunks of audio files into folders of corresoonding phonemes. TIMIT corpus already has phoneme segmentation\
_timit_parser_results.zip_ contains result output of the script above.\
### 5. ML moder training
_train.py_ tries to do that. Uses _librosa_ and _keras_\
The basis of this algorithm is based on this [article](https://towardsdatascience.com/speech-classification-using-neural-networks-the-basics-e5b08d6928b7), but significant changes were made.\
There are different modes. Each mode teaches the model based by different metrics, just change the variable `mode` in the code. With very small amount of data, on four phonemes and ten epochs, the accuracy is as follows:\
0: 28 % (don't use this mode, it basically feeds the model the raw audio, i. g the amplitude over time)\
1: 68 %\
2: 50 %\
3: 68 %\
When run mode 1 on dataset of all phonemes (60 of them) for 25 epochs, the accuracy was 45 % and still not converging, so I guess with more epochs (in terms of order of magnitude), we could achive decent results.
### Future developement
* Find a way to use more of these metrics at once.\
* Primarily, find how not use all the data currently used, for example extract only some data from the spectrum, like the most prominent overtones and feed the model only this. Also try to find other metrics, compile all this together and teach the model on that. The biggest problem now is scalabiity because of memory. The spectrum itself is really huge to be used whole, so feeding the model with larger set of data is currently impossible on common GPUs.\
* Once done the above, try fiddling with the window parameter of short time FT (parameter n_fft of librosa.stft())\
* Make a better selection of phonemes. Timit uses something around 60, which is way more than needed (current sets have around 40 for English). There are also phonemes which differ only in length and thats context dependent, therefore indistinguishible for the model.

### Blind paths
_pyAudioAnalysis_ tool is not that much customizable and doesn't acceps audio files as short as we need.
### Notes
Those people do something similar to what we do: https://easypronunciation.com/en/
### References
Garofolo, J. & Lamel, Lori & Fisher, W. & Fiscus, Jonathan & Pallett, D. & Dahlgren, N. & Zue, V.. (1992). TIMIT Acoustic-phonetic Continuous Speech Corpus. Linguistic Data Consortium.\
https://towardsdatascience.com/speech-classification-using-neural-networks-the-basics-e5b08d6928b7 \
