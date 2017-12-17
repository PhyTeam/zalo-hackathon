from gtts import gTTS
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt

def FFTND(sound):
    data = AudioSegment.get_array_of_samples(sound)
    sampling_length = sound.frame_rate # measured every 15 minutes
    Fs = 1.0 / sampling_length
    ls = range(len(data))  # data contains the function
    freq = np.fft.fftfreq(len(data), d=sampling_length)
    fft = np.fft.fft(data)
    x = freq[:len(data) / 2]
    for i in range(len(x)):
        if x[i] > 0.005:  # cut off all frequencies higher than 0.005
            fft[i] = 0.0
            fft[len(data) / 2 + i] = 0.0
    inverse = np.fft.ifft(fft)
    #
    # try:
    #     plt.clf()
    #     plt.figure(2)
    #     plt.subplot(311)
    #     plt.title("Origin")
    #     plt.plot(inverse, 'b')
    #
    #     plt.show()
    # except Exception as e:
    #     print("plotting exception")
    #     print(str(e))

    return sound._spawn(inverse.tolist())

sound = AudioSegment.from_file("ckta-o.mp3", format="mp3", frame_rate=16000, channels=2)
# result = FFTND(sound)

# result.export("output.mp3", format="mp3")
[left, right] = sound.split_to_mono()
right = AudioSegment.invert_phase(right)
instrument = left.overlay(right)
invert_instrument = AudioSegment.invert_phase(instrument)
vocals = sound.overlay(invert_instrument)
#vocals.export("output.mp3", format="mp3")
try:
    plt.clf()
    plt.figure(2)
    plt.subplot(311)
    plt.title("Origin l")
    plt.plot(sound.get_array_of_samples()[:5000000], 'b')

    plt.subplot(312)
    plt.title("Instrument")
    plt.plot(instrument.get_array_of_samples()[:5000000], 'b')
    plt.subplot(313)
    plt.title("Origin 2")
    plt.plot(right.get_array_of_samples()[:5000000], 'b')

    plt.show()
except Exception as e:
    print("plotting exception")
    print(str(e))