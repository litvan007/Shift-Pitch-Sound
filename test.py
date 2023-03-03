# -*- coding: utf-8 -*-
from phase_decoder import Phase_decoder
from scipy.io import wavfile
import numpy as np
import sys

if __name__ == '__main__':
    input_file_name, output_file_name, time_stretch_ratio = sys.argv[1:]
    samplerate, data = wavfile.read(input_file_name)

    # для частоты дискретизации наилучшим вариантов выбора длины окна будет 512, с шагом 128

    alpha = 1 / float(time_stretch_ratio)

    decoder = Phase_decoder(samplerate=samplerate, signal=data, frame_duration=0.032, hop_duration=0.008, alpha=alpha)
    pitch_shifted_signal = decoder.pitch_shift()

    wavfile.write(output_file_name, samplerate, pitch_shifted_signal.astype(np.int16))
