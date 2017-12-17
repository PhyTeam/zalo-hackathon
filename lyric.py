import pickle
from tempfile import TemporaryFile
#from pylab import int16, fft
from scipy.io import wavfile
from pydub import AudioSegment
import re
from gtts import gTTS
import os.path
from peek_lyric import detect, barrier
import numpy as np

song_name = "ne"
path = "data/" + song_name
soundtrack = AudioSegment.from_file(path + ".mp3", format="mp3", channels=2)
fs = soundtrack.frame_rate
[left, right] = soundtrack.split_to_mono()
right = AudioSegment.invert_phase(right)
instrument = left.overlay(right)
instrument = AudioSegment.from_mono_audiosegments(instrument, instrument)
instrument.export("instrument.mp3", format="mp3")

lmap = []
word_dict = dict()

if os.path.isfile("cacheword.pkl"):
    with open("cacheword.pkl", "rb") as f:
        word_dict = pickle.load(f)
snd_o = soundtrack.get_array_of_samples()
snd_o = np.asarray(snd_o, dtype=np.float32)
snd_o = snd_o / (max(snd_o.max(), abs(snd_o.min())))
word = detect(snd_o)
vocal_flags = barrier(soundtrack, instrument)
new_word = []
for k in word:
    k = int(k)
    if (vocal_flags[min(k, len(vocal_flags) - 1)]):
        new_word.append(k)
word = new_word

count_lw = 0
with open(path+".lrc", "rb") as lf:
    lf = lf.read()
    data = lf.decode().split("\n")
    for line in data:
        match = re.search("\[(\d\d):(\d\d).(\d\d)\](.*)", line)
        if match != None:
            m, s, ms, str_line = match.groups()
            idx = int((int(m)*60 + int(s) + int(ms)/100)*fs*2)
            str_line = str_line.replace(".", " ").replace(",", " ").replace("\r", " ")
            sound_arr = []
            word_array = str_line.split()
            count_lw += len(word_array)
            for iw in word_array:
                iw = iw.lower()
                w_sound = word_dict.get(iw, None)
                if w_sound == None:
                    print("cache " + str(iw))
                    tts = gTTS(text=iw, lang="vi")
                    tts.save("tmp.mp3")
                    w_sound = AudioSegment.from_file(file="tmp.mp3", format="mp3")
                    word_dict[iw] = w_sound
                sound_arr.append(w_sound)
            if len(sound_arr) != 0:
                lmap.append((idx, sound_arr))
with open("cacheword.pkl", "wb+") as f:
    pickle.dump(word_dict, f)
print("Num peek Word: " + str(len(word)))
print("Num lyric Word: " + str(count_lw))

max_index = len(soundtrack)
for i, l in enumerate(lmap):
    count_ol = 0
    print("Processing ..." + str(i) + "/" + str(len(lmap)), end=",")
    index, sound_arr = l
    _a = index
    _b = lmap[i + 1][0] if i + 1 < len(lmap) else max_index
    pos = []
    for x in word:
        if x >= _a and x <= _b:
            pos.append(x)

    for y in range(len(pos)):
        milisecond = pos[y]//96
        instrument = instrument.overlay(sound_arr[min(y, len(sound_arr) - 1)], position=milisecond)
        count_ol += 1
    print(count_ol, len(sound_arr))
instrument.export("final.mp3", format="mp3")