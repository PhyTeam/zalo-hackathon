from pylab import int16, fft
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import math

import onset_detection

trim_threshold = 10

def trim_wave(data):
    i = 0
    for i in range(len(data)):
        if data[i] > trim_threshold:
            break
    return np.asarray(data[i:], dtype=np.float32)

def fft_display(data, sampFreq):
    n = len(data)
    p = fft(data)
    nUniquePts = int(math.ceil((n + 1) / 2))
    p = p[0:nUniquePts]
    p = abs(p)

    p = p / float(n)
    p = p ** 2  # square it to get the power

    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if n % 2 > 0:  # we've got odd number of points fft
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) - 1] = p[1:len(p) - 1] * 2  # we've got even number of points fft

    freqArray = np.arange(0, nUniquePts, 1.0) * (sampFreq / n);
    plt.figure(1)
    plt.plot( freqArray, 10 * np.log10(p), color='k')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power (dB)')

def show_waveform(w, title, subplot):
    plt.figure(2)
    plt.subplot(subplot)
    plt.title(title)
    plt.plot(w, 'b')

def smooth_wave(wave):
    wn_size = 4
    smooth_wave = []
    for i in range(len(wave)):
        smooth_wave.append(sum(wave[min(0, i - wn_size / 2): (i + wn_size / 2)]) / wn_size)
    return np.array(smooth_wave)


def detect(sound):
    waveform = sound[:]
    waveform = np.abs(waveform)
    #show_waveform(waveform, "Squared wave", 311)

    # sw = smooth_wave(waveform)
    # show_waveform(sw, "Smoothed wave", 312)
    # plt.show()
    candidates = onset_detection.detect_onsets(waveform)
    print(candidates)

    # dis = np.zeros(waveform.shape)
    # dis -= 1
    # for i in candidates:
    #     dis[i] = 1
    # plt.plot(dis.tolist(), 'g')
    return candidates


def barrier(soundtrack, background):
    snd_o = np.array(soundtrack.get_array_of_samples())
    snd_b = np.array(background.get_array_of_samples())

    snd_o = snd_o / (max(snd_o.max(), abs(snd_o.min())))
    snd_b = snd_b / (max(snd_b.max(), abs(snd_b.min())))

    #detect(snd_o[:1000000])

    sa = []
    step = 5000
    #print(sampFreq_o)
    for i in range(0, len(snd_o), step):
        max_idx = min(i+step, len(snd_o) - 1)
        omax = snd_o[i: max_idx].max()
        omin = snd_o[i: max_idx].min()

        #print(i, max_idx)
        bmax = snd_b[i: max_idx].max()
        bmin = snd_b[i: max_idx].min()

        zmax = snd_o[max(i - step, 0): min(i + 2 * step, len(snd_o) - 1)].max()
        zmin = snd_b[max(i - step, 0): min(i + 2 * step, len(snd_o) - 1)].min()

        ret = abs((omax - omin) - (bmax - bmin) / (zmax / zmax))
        sa.append([ret]*step)

    sa = np.asarray(sa).reshape((len(sa)*step, 1))
    #print(sa)
    return sa > 0.5
# [ snd_o[i: i+100].max() - snd_o[i: ] for i in range(0, size - 100, 100)]

#voice = np.asarray([b - a for a, b in zip(snd_b[0:size], snd_o[0:size])])
#wavfile.write("voice.wav", sampFreq_o, voice)

#snd = snd / (2.**15)


#
# try:
#     #plt.clf()
#     plt.figure(2)
#     plt.subplot(311)
#     plt.title("Origin")
#     plt.plot(snd_o[0:size], 'b')
#     plt.plot([0 if i < 0 or i > len(sa) else 1 if sa[i] > 0.5 else -1 for i in range(-step, len(sa))], 'r')
#
#     plt.subplot(312)
#     plt.title("Beat")
#     plt.plot(snd_b[0:size], 'b')
#
#     plt.subplot(313)
#     plt.title("Sub")
#     plt.plot(sa, 'r')
#
#     plt.show()
# except Exception as e:
#     print("plotting exception")
#     print(str(e))
