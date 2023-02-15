# Whisper-Speech-To-Text
This is an implementation of OpenAI's Whisper for the purpose of speech-to-text via default microphone, enabling direct output to clipboard.

## Installation
Installation is as easy as: 

`pip install -U openai-whisper`

`pip install -r requirements.txt`

I also strongly encourage the installation of PyTorch with cuda.

## Usage
Most input arguments carry over from the base Whisper package.

Those of note are:

------------------

`--model`			 default is the 'small' model, some others are the 'tiny', 'base', 'medium', and 'large' models

`--model_dir`		 default is set as None, 'Users/[username]/.cache/whisper' is used as the default dir

`--device`		 default is 'cuda', set this to 'cpu' if you don't have cuda installed with torch

`--task`			 default is transcribe (duh), but translation is also possible

`--language`		 default is set as None (enabling language detection), but I recommend setting this to English/French/Spanish/etc. to cut down on inference times

`--temperature`		 default is 0, depolarizes output distribution allowing for more 'creativity'

`--threads`		 default is 0, this refers to CPU threads

----------------------

Novel input arguments:

----------------------

`--amplification`		 default is 1.0, amplifies the recording by a given floating point multiple

`--print_text`		 default is not set/False, prints text to your CLI/terminal

`--no_clipboard`		 default is not set/False, prevents output from being sent to your clipboard

`--hotkey`		 default is 'ctrl+shift+r', the combination that activates audio recording and inferencing
