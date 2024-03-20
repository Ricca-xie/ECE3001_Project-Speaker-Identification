import librosa
import numpy as np
import time
import librosa.display
import math
import matplotlib.pyplot as plt

def extract_hpss_features_sg(wav_path, max_length, window_length=320, window_shift=160, use_stft=True):
    """Extract Harmonic-Percussive Source Separation features.

    Args:
      wav_dir: string, directory of wavs.
      out_dir: string, directory to write out features.
      recompute: bool, if True recompute all features, if False skip existed
                 extracted features.
    """
    cnt = 0
    t1 = time.time()
    (audio, sr) = read_audio(wav_path)

    if audio.shape[0] == 0:
        print("File %s is corrupted!" % wav_path)
        raise ValueError
    else:
        # librosa.display.waveshow(audio, sr=sr)
        # plt.show()

        if use_stft: # compute stft
            spec = np.log(get_spectrogram(audio, window_length, window_shift) + 1e-8)
        else: # not use stft
            frame = 256
            split_num = math.floor(audio.shape[0] / frame)
            new_audio = np.split(audio[:split_num*frame], split_num)
            spec = np.stack(new_audio, axis=0).T

        spec = norm(spec)
        spec = spec.T
        spec = pad_trunc_seq(spec, max_length)

        # cnt += 1
    # print("Thread %d Extracting feature time: %s" % (i, (time.time() - t1)))
    return spec

def read_audio(path, target_fs=None):
    try :
        audio, fs = librosa.load(path, sr=None) # fs:sample rate
    except:
        print(path)

    if audio.ndim > 1:  # 维度>1，这里考虑双声道的情况，维度为2，在第二个维度上取均值，变成单声道
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)  # 重采样输入信号，到目标采样频率
        fs = target_fs
    return audio, fs

def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length.

    Args:
      x: ndarray, input sequence data.
      max_len: integer, length of sequence to be padded or truncated.

    Returns:
      ndarray, Padded or truncated input sequence data.
    """
    L = len(x)
    shape = x.shape
    if L < max_len:
        pad_shape = (max_len - L,) + shape[1:]
        pad = np.zeros(pad_shape)
        x_new = np.concatenate((x, pad), axis=0)
    else:
        x_new = x[0:max_len]

    return x_new

def get_spectrogram(wav, win_length, win_shift):
    D = librosa.stft(wav, n_fft=win_length, hop_length=win_shift, win_length=win_length, window='hamming')
    spect, phase = librosa.magphase(D)
    return spect


def norm(spec):
    mean = np.reshape(np.mean(spec, axis=1), (spec.shape[0],1))
    std = np.reshape(np.std(spec, axis=1), (spec.shape[0],1))
    spec = np.divide(np.subtract(spec,np.repeat(mean, spec.shape[1], axis=1)), np.repeat(std, spec.shape[1], axis=1))
    return spec




