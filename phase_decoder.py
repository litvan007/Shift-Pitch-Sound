# -*- coding: utf-8 -*-
import numpy as np

from scipy.fft import fft, ifft
from scipy.signal import hann
from scipy.interpolate import interp1d

class Phase_decoder:
    def __init__(self, samplerate, signal, frame_duration, hop_duration, alpha):
        self.samplerate = samplerate
        self.signal = signal
        self.n = len(signal)

        self.window_size = int(samplerate * frame_duration)
        self.hop_size = int(samplerate * hop_duration)
        # self.alpha = 2 ** (semitones / 12) # scaling factor
        self.alpha = alpha

    @staticmethod
    def create_frames(x, hop_size, window_size):
        n = len(x)

        frames = []
        for i in range(0, n - window_size, hop_size):
            frame = x[i:i+window_size]
            frames.append(frame)
        return np.array(frames)

    @staticmethod
    def fusion_frames(frames, hop_size):
        n_frames = frames.shape[0]
        size_frames = frames.shape[1]

        vector_time = np.zeros(n_frames * hop_size - hop_size + size_frames)

        time_i = 0

        for i in range(n_frames):
            vector_time[time_i:time_i + size_frames] = vector_time[time_i:time_i+size_frames] + frames[i]

            time_i = time_i + hop_size

        return vector_time 
            

    def pitch_shift(self):
        hop_out = round(self.alpha*self.hop_size)
        wn = hann(self.window_size * 2 + 1) # окно ханна для сглаживания фрейма 
        wn = wn[1:self.window_size+1]

        x = self.signal
        x = np.concatenate((np.squeeze(np.zeros((self.hop_size*3, 1))), x), axis=0)
        frames = self.create_frames(x, self.hop_size, self.window_size)        

        output = np.zeros((len(frames), self.window_size))

        phase_cumulative = 0
        previous_phase = 0

        for i in range(len(frames)):

        # Analysis
            curr_frame = frames[i]
            curr_frame_win = curr_frame * wn / np.sqrt(self.window_size / (self.hop_size * 2))

            curr_frame_win_fft = fft(curr_frame_win)
            magnitude_frame = np.abs(curr_frame_win_fft)
            phase_frame = np.angle(curr_frame_win_fft)


        # Processing
            delta_phi = phase_frame - previous_phase 
            previous_phase = phase_frame 

            delta_phi_prime = delta_phi - self.hop_size * 2*np.pi*np.arange(self.window_size)/self.window_size 
            delta_phi_prime_mod = np.mod(delta_phi_prime+np.pi, 2*np.pi) - np.pi 

            true_freq_frame = 2*np.pi*np.arange(self.window_size)/self.window_size + delta_phi_prime_mod/self.hop_size
            phase_cumulative = phase_cumulative + hop_out * true_freq_frame 

        # Synthesis
            output_frame = np.real(ifft(magnitude_frame * np.exp(1j*phase_cumulative)))
            output[i] = output_frame * wn / np.sqrt(self.window_size / hop_out)/2

    # Final (склеика фреймов c интерполяцией для восстановления исходной длины сигнала)
        output_time_changed = self.fusion_frames(output, hop_out)
        output_time = interp1d(np.arange(len(output_time_changed)), output_time_changed, kind='linear')(np.arange(0, len(output_time_changed)-1, self.alpha))

        return output_time







        

