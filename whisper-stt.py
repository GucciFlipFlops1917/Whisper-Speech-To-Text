import keyboard
import pyaudio
import pyperclip
import numpy as np
from scipy.io import wavfile
import os.path
import argparse
import warnings
import torch
from whisper import load_model, available_models
from whisper.transcribe import transcribe
from whisper.model import Whisper
from whisper.utils import str2bool, optional_int, optional_float
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
parser.add_argument("--verbose", type=str2bool, default=False, help="whether to print out the progress and debug messages")
parser.add_argument("--amplification", type=float, default=1.0, help="inference audio integer amplification/boost")
parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")
parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")
parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")
parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
parser.add_argument("--print_text", action='store_true', default=False, help="print text to CLI")
parser.add_argument("--no_clipboard", action='store_true', default=False, help="don't send text to clipboard")
parser.add_argument("--hotkey", type=str, default='ctrl+shift+r', help="key combination to activate inference")
args = parser.parse_args().__dict__

hotkey = args.pop("hotkey")
print_terminal = args.pop("print_text")
no_clipboard = args.pop("no_clipboard")
model_name: str = args.pop("model")
model_dir: str = args.pop("model_dir")
amplification = args.pop("amplification")
device: str = args.pop("device")

if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
    if args["language"] is not None:
        warnings.warn(f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead.")
    args["language"] = "en"
temperature = args.pop("temperature")
if (increment := args.pop("temperature_increment_on_fallback")) is not None:
    temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
else:
    temperature = [temperature]
if (threads := args.pop("threads")) > 0:
    torch.set_num_threads(threads)
model = load_model(model_name, device=device, download_root=None)

class Recorder():
    def __init__(self):
        self.sample_rate = 44100
        self.chunk = 4096
        self.filename = 'output.wav'
        while os.path.exists(self.filename):
            self.filename = str('1_')+self.filename
    def record(self):
        print("Recording started")
        recorded_data = []
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1,
                        rate=44100, input=True,
                        frames_per_buffer=self.chunk)
        while(True):
            data = stream.read(self.chunk)
            recorded_data.append(data)
            if keyboard.is_pressed(hotkey):
                print("Recording stopped")
                # stop and close the stream
                stream.stop_stream()
                stream.close()
                p.terminate()
                #convert recorded data to numpy array
                recorded_data = [np.frombuffer(frame, dtype=np.float32) for frame in recorded_data]
                wav = np.concatenate(recorded_data, axis=0)
                wav = wav*amplification
                wavfile.write(self.filename, 44100, wav)
                print(f"Wavefile temp written as '{self.filename}'")
                result = transcribe(model, self.filename, temperature=temperature, **args)
                if not result['text']=='':
                    if not no_clipboard==True:
                        pyperclip.copy(result['text'])
                        print('Copied to clipboard')
                    if print_terminal==True:
                        print(result['text'])
                else:
                    print('No text detected!')
                os.remove(self.filename)
                print(f"Waiting for '{hotkey}' to inference")
                break

recorder = Recorder()
print(f"Waiting for '{hotkey}' to inference")
keyboard.add_hotkey(hotkey, recorder.record)
keyboard.wait()
