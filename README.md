# Speech Emotion Recognition using the CNN and CRNN architecture

Daily the amount of audio data related to feedback evaluation increases, especially considering business on digital platforms. Since through audio it's possible to directly identify characteristics as sentiments, irony, sarcasm and others that could be impossible only by text it is really important to the companies to consider analysing it to get some insigths. Despite that the higher amount of data and the human bias can make mistakes not getting what the customers really wanted to express. So, in this work it's used some techniques of machine learning to classify sentiments on audio speech.

It was applied the dataset [TESS Toronto emotional speech set](https://github.com/gabrfern99/speech-sentanalysis/tree/main/TESS%20Toronto%20emotional%20speech%20set%20data) and two architectures, CNN and CRNN. 
Follow the instructions to use them.

## Installation

### Clone the Repository
```
git clone https://github.com/gabrfern99/speech-sentanalysis
cd speech-sentanalysis
```

### Install the Required Packages

```
pip install -r requirements.txt
```

### Run the code

To use the first code:
```
Emotional_Speech_Recognition - TESS.ipynb
```

To use the second one:
```
python3 crnn_tess_sentanalysis.py
```
