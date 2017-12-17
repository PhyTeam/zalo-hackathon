import pickle

from gtts import gTTS
import os
from pydub import AudioSegment
if os.path.isfile("cacheword.pkl"):
    with open("cacheword.pkl", "rb") as f:
        word_dict = pickle.load(f)

# tts = gTTS(text='nhẹ', lang='vi')
# tts.save("good.mp3")
# sound = AudioSegment.from_mp3("good.mp3")
sound = word_dict.get("tim", None)
background = AudioSegment.from_mp3("ckta-o.mp3")
[left, right] = background.split_to_mono()
right = AudioSegment.invert_phase(right)
instrument = left.overlay(right)
background = instrument.overlay(sound, position=6000)

sound = word_dict.get("chạm", None)
background = background.overlay(sound, position=10000)
background.export("test.mp3", format="mp3")
#os.system("mpg321 good.mp3")