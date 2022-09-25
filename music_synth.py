#!/usr/bin/env python
# coding: utf-8

# # Signal Processing (Analysis of musical tunes) - Term Project

# # Importing required libraries...

# In[9]:


import IPython
from scipy.io import wavfile
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
import random
from pydub import AudioSegment
from pydub.playback import play
import thinkdsp


# In[10]:

noteFreqs={
    "C0":16.35 , "Db0":17.32 , "D0":18.35 ,"Eb0":19.45,
    "E0" : 20.60, "F0" : 21.83, "Gb0" : 23.12, "G0" : 24.50,
    "Ab0" : 25.96, "A0" : 27.50, "Bb0" : 29.14, "B0" : 30.87,
    "C1" : 32.70, "Db1" : 34.65, "D1" : 36.71, "Eb1" : 38.89,
    "E1" : 41.20, "F1" : 43.65, "Gb1" : 46.25, "G1" : 49.00,
    "Ab1" : 51.91, "A1" : 55.00, "Bb1" : 58.27, "B1" : 61.74,
    "C2" : 65.41, "Db2" : 69.30, "D2" : 73.42, "Eb2" : 77.78,
    "E2" : 82.41, "F2" : 87.31, "Gb2" : 92.50, "G2" : 98.00,
    "Ab2" : 103.83, "A2" : 110.00, "Bb2" : 116.54, "B2" : 123.47,
    "C3" : 130.81, "Db3" : 138.59, "D3" : 146.83, "Eb3" : 155.56,
    "E3" : 164.81, "F3" : 174.61, "Gb3" : 185.00, "G3" : 196.00,
    "Ab3" : 207.65, "A3" : 220.00, "Bb3" : 233.08, "B3" : 246.94,
    "C4" : 261.63, "Db4" : 277.18,"D4" : 293.66, "Eb4" : 311.13,
    "E4" : 329.63, "F4" : 349.23, "Gb4" : 369.99, "G4" : 392.00,
    "Ab4" : 415.30, "A4" : 440.00, "Bb4" : 446.16, "B4" : 493.88,
    "C5" : 523.25, "Db5" : 554.37, "D5" : 587.33, "Eb5" : 622.25,
    "E5" : 659.25, "F5" : 698.46, "Gb5" : 739.99, "G5" : 783.99,
    "Ab5" : 830.61, "A5" : 880.00, "Bb5" : 932.33, "B5" : 987.77,
}
randomCoeffs = []
for i in range(0, 8):
    randomCoeffs.append(random.uniform(-1, 1))
    
fourierCoeffs ={
    "sine": [0, 1, 0, 0, 0, 0, 0, 0],
    "sawtooth": [0, 0.6366, 0, -0.2122, 0, 0.1273, 0, -0.0989],
    "trumpet": [0.1155, 0.3417, 0.1789, 0.1232, 0.0678, 0.0473, 0.0260, 0.0045, 0.0020],
    "random": randomCoeffs
}


def createNote(noteName="A4", type="sine", amp=0.5, beats=1.0, filter=None,cutoff=None,filename="defaultFile"):
# Initialize some values, let signal be empty first
    frequency = noteFreqs[noteName]
    duration = beats / 2
    signal = thinkdsp.SinSignal(freq=0)
# Add harmonics to the signal according to their Fourier Synthesis Coefficients
    for i in range(0, 8):
        signal += thinkdsp.SinSignal(freq=frequency*i, amp=amp*fourierCoeffs[type][i], offset=0)
# Convert signal into wave to .wav file to AudioSegment to be mixed and played by the program
    wave = signal.make_wave (duration=duration, start=0, framerate=44100)
   # wave.plot()
    wave.write(filename=filename)
    audio = AudioSegment.from_wav(filename)
    print("Creating note " + noteName + " at " + str(frequency) + " for " + str(beats) + " beats with the ")
# Add filters if necessary
    if filter == "lowPass":
        audio = audio.low_pass_filter(cutoff)
        print("Applying Low-Pass Filter")
    if filter == "highPass":
        audio = audio.high_pass_filter(cutoff)
        print("Applying High-Pass Filter")
    return audio
  

def createSpace(track, attack=100, release=100):
    for i in range(0,len(track)-1):
        if track[i][0:2]==track[i+1][0:2] :
            track[i]= track[i].fade_out(duration=release)
        else:
            return None

# Combines two audio tracks
def mix2tracks(track1, track2):
    createSpace(track1, attack=50, release=50)
    createSpace(track2, attack=50, release=50)
    song = AudioSegment.empty()
    for i in range(len(track1)):
        note1 = track1[i]
        note2 = track2[i]
        song += note1[:len(note1)].overlay(note2[:len(note2)])
        
    return song

# Create notes for 1st song with the standard Sinewave synthesizer
# Default: quarter note, Long: half note
G3_long= createNote("G3",type= "sine", beats=2.0)
C4 = createNote("C4", "sine")
D4 = createNote("D4", "sine"); D4_long= createNote("D4", "sine", beats=2.0)
Eb4 = createNote("Eb4", "sine")
E4 = createNote("E4", "sine")
F4_long= createNote("F4", "sine", beats=2.0)
Gb4 = createNote("Gb4", "sine"); Gb4_long = createNote("Gb4", "sine", beats=2.0)
G4 = createNote("G4", "sine"); G4_long= createNote("G4", "sine", beats=2.0)
Ab4 = createNote("Ab4", "sine");
A4 = createNote("A4", "sine"); A4_long= createNote("A4", "sine", beats=2.0)
B4 = createNote("B4", "sine"); B4_long= createNote("B4", "sine", beats=2.0)
C5 = createNote("C5", "sine")
D5 = createNote("D5", "sine"); D5_long= createNote("D5", "sine", beats=2.0)
G5_long= createNote("G5", "sine", beats=2.0)

# Song 1: Jingle Bells
track1 = [B4, B4, B4_long, B4, B4, B4_long, B4, D5, G4, A4, B4_long, B4_long,
          C5, C5, C5, C5, C5, B4, B4, B4, B4, A4, A4, B4, A4_long, D5_long,
          B4, B4, B4_long, B4, B4, B4_long, B4, D5, G4, A4, B4_long, B4_long,
          C5, C5, C5, C5, C5, B4, B4, B4, D5, D5, C5, A4, G4_long, G5_long]

track2 = [G4, B4, D4_long, G4, B4, D4_long, G4, B4, D4, G4, G4_long, F4_long,
          E4, G4, Eb4, G4, D4, G4, E4, Ab4, A4, E4, C4, E4, D4_long, Gb4_long,
          G4, B4, D4_long, G4, B4, D4_long, G4, B4, D4, G4, G4_long, F4_long,
          E4, G4, Eb4, G4, D4, G4, E4, Ab4, A4, E4, D4, Gb4, G4_long, G3_long]

song1 = mix2tracks(track1=track1, track2=track2)

print("\n *** NOW PLAYING *** In Jingle Bells on the SineWave Synthesizer!")

play(song1)

wav_loc = "Random.wav"
rate, data = wavfile.read(wav_loc)
data = data / 32768


# In[13]:


def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    return fftnoise(f)


# In[14]:

IPython.display.Audio(data=data, rate=rate)

# # Plotting the Audio signal

# In[ ]:


fig, ax = plt.subplots(figsize=(20,4));
ax.plot(data);


# # Adding noise to the Audio clip

# In[15]:


noise_len = 2 # seconds
noise = band_limited_noise(min_freq=4000, max_freq = 12000, samples=len(data), samplerate=rate)*10
noise_clip = noise[:rate*noise_len]
audio_clip_band_limited = data+noise

fig, ax = plt.subplots(figsize=(20,4))
ax.plot(audio_clip_band_limited,color='black' )
IPython.display.Audio(data=audio_clip_band_limited, rate=rate)


# # Denoising the audio signal 

# The Short-time Fourier transform (STFT), is a Fourier-related transform used to determine the sinusoidal frequency and phase content of local sections of a signal as it changes over time.

# Procedure for computing STFTs is to divide a longer time signal into shorter segments of equal length and then compute the Fourier transform separately on each shorter segment

# In[10]:


import time
from datetime import timedelta as td


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def plot_spectrogram(signal, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_statistics_and_filter(
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
    plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
    plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("Threshold for mask")
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask")
    plt.show()


def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0,
    verbose=False,
    visual=False,
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """
    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    if visual:
        plot_spectrogram(noise_stft_db, title="Noise")
    if visual:
        plot_statistics_and_filter(
            mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
        )
    if visual:
        plot_spectrogram(sig_stft_db, title="Signal")
    if visual:
        plot_spectrogram(sig_mask, title="Mask applied")
    if visual:
        plot_spectrogram(sig_stft_db_masked, title="Masked signal")
    if visual:
        plot_spectrogram(recovered_spec, title="Recovered spectrogram")
    return recovered_signal


# In[11]:


output_plot = removeNoise(audio_clip=audio_clip_band_limited, noise_clip=noise_clip,verbose=True,visual=True)


# In[16]:


fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
plt.plot(output_plot, color='red')
ax.set_xlim((0, len(output_plot)))
plt.show()
# play back a sample of the song
IPython.display.Audio(data=output_plot, rate=44100)


# In[ ]:


