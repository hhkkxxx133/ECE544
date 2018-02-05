# feature.py
# Created by dwang49
# Implement several methods to extract features from plain monotone audio data.
# Methods include plain audio slicing(no extraction), short time fourier transform,
# Filter banks and MFCC 
# References:
# http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
# https://github.com/buriburisuri/speech-to-text-wavenet
# https://github.com/jameslyons/python_speech_features


import numpy as np
from scipy.fftpack import dct
from scipy.signal import stft


from config import feature_config


# Feature generator
class feature_generator():

    # Sanity check and loggin
    def __init__(self):

        assert feature_config.feature_method in ['time', 'stft',
                                                 'fbank', 'mfcc']
        self.feature_method = feature_config.feature_method

        self.fs = int(feature_config.fs)

        assert feature_config.pre_emphasis_coef in (0, 0.95, 0.97)
        self.pre_emphasis_coef = feature_config.pre_emphasis_coef

        self.frame_length = int(feature_config.frame_length
                                * feature_config.fs / 1000)

        self.overlap_length = int(feature_config.overlap_length
                                  * feature_config.fs / 1000)

        assert feature_config.nfft in (256, 512, 1024)
        self.nfft = feature_config.nfft

        self.nfbanks = int(feature_config.nfbanks)

        self.nmfccs = int(feature_config.nmfccs)

        self.norm_mean = bool(feature_config.norm_mean)

        assert feature_config.out_dtype in (np.float, np.float32)
        self.out_dtype = feature_config.out_dtype

        print('\n--------------------------------------------------')
        print('Feature extraction configurations:')
        print('>>> Method: {}'.format(self.feature_method))
        print('>>> Sampling rate: {} Hz'.format(self.fs))
        print('>>> Pre emphasis coefficient: {}'.format(
            self.pre_emphasis_coef))
        print('>>> Frame length: {} milliseconds <-> {} samples'.format(
            feature_config.frame_length, self.frame_length))
        print('>>> Overlapping length: {} milliseconds <-> {} samples'.format(
            feature_config.overlap_length, self.overlap_length))
        print('>>> FFT Length: {}'.format(self.nfft))
        print('>>> Filter banks number: {}'.format(self.nfbanks))
        print('>>> MFCC coefficients number: {}'.format(self.nmfccs))
        print('>>> Mean removal normalization: {}'.format(self.norm_mean))
        print('>>> Output data type: {}'.format(self.out_dtype))
        print('--------------------------------------------------\n')

        return

    # Feature extraction method: 'main'
    # Wrapper around each method
    # Output: np.float32 array in [Batch X NFrames X NData]
    def feature_extract(self, data):

        # Apply pre-processing
        data_pro = self.pre_processing(data)

        # Jump to target methods
        method_func = 'feature_' + self.feature_method
        feature = getattr(self, method_func)(data_pro)

        # Mean removal normalization if applicable
        if self.norm_mean:
            feature = self.remove_mean(feature)

        # Deal with dtype and transpose
        ret = np.transpose(feature.astype(feature_config.out_dtype),
                           axes=(0, 2, 1))

        return ret

    # Feature extraction method: 'time'
    # Break down time domain signals into overlapping frames
    # Output: np.float32 array in [Batch X NData X NFrame]
    def feature_time(self, data):

        # Zero-padding at start and end of the array
        zeros = np.zeros((data.shape[0], self.frame_length // 2),
                         dtype=data.dtype)
        data_pad = np.concatenate((zeros, data, zeros), axis=1)

        # Locate starting point
        sidx = np.arange(0, data.shape[1] + 1,
                         self.frame_length - self.overlap_length)

        # Container
        ret = np.zeros((data.shape[0], self.frame_length, sidx.shape[-1]),
                       dtype=self.out_dtype)

        # Extract Frame
        for fidx in range(sidx.shape[-1]):
            ret[:, :, fidx] = data_pad[:, sidx[fidx]: sidx[fidx]
                                       + self.frame_length]

        return ret

    # Feature extraction method: 'stft'
    # Basically just do stft
    # Output: np.float32 array in [Batch X NData X NFrame]
    def feature_stft(self, data):

        # Power Spectrum
        power_spec = self.power_stft(data)

        # Convert to dB
        dB_spec = self.num2dB(power_spec)

        return dB_spec

    # Feature extraction method: 'fbank'
    # Power spectrum and apply
    # Output: np.float32 array in [Batch X NData X NFrame]
    def feature_fbank(self, data):

        # Power Spectrum
        power_spec = self.power_stft(data)

        # Apply filter banks
        filter_spec = self.apply_fbanks(power_spec)

        # Convert to dB
        dB_spec = self.num2dB(filter_spec)

        return dB_spec

    # Feature extraction method: 'mfcc'
    # Filter banks with DCT
    # Output: np.float32 array in [Batch X NData X NFrame]
    def feature_mfcc(self, data):

        # Power Spectrum
        power_spec = self.power_stft(data)

        # Apply filter banks
        filter_spec = self.apply_fbanks(power_spec)

        # Convert to dB
        dB_spec = self.num2dB(filter_spec)

        # Apply DCT and keep 2-13
        mfcc = dct(dB_spec, type=2, axis=1,
                   norm='ortho')[:, 1: (self.nmfccs + 1), :]

        # Construct sinusoidal liftering
        ncoeffs = mfcc.shape[1]
        n = np.arange(ncoeffs)
        lift = 1 + (22 / 2) * np.sin(np.pi * n / 22)

        # Apply sinusoidal liftering
        trans_mfcc = np.transpose(mfcc, axes=(0, 2, 1))
        lift_trans_mfcc = np.matmul(trans_mfcc, np.diag(lift))
        lift_mfcc = np.transpose(lift_trans_mfcc, axes=(0, 2, 1))

        return lift_mfcc

    # Time Domain Pre-processing
    # Output: np.int16 array in [Batch X NData]
    def pre_processing(self, data):

        # Do not modify
        if self.pre_emphasis_coef == 0:
            return data

        # Pre Emphasis Operation x[t] = x[t] - a*x[t-1]
        ret = np.concatenate(
            (np.expand_dims(data[:, 0], 1),
             data[:, 1:] - data[:, :-1] * self.pre_emphasis_coef),
            axis=1)

        return ret

    # Power spectrum (periodgram) implementation
    # stft and converting to power
    # Output: np.float32 array in [Batch X NData X NFrame]
    def power_stft(self, data):

        # stft with zero padding, hamming window and overlapping
        _, _, spec = stft(data, fs=self.fs, window='hamming',
                          nperseg=self.frame_length, nfft=self.nfft,
                          noverlap=self.overlap_length)

        # Power spectrum (periodogram)
        power_spec = np.square(np.absolute(spec, dtype=self.out_dtype))
        power_spec /= self.nfft

        return power_spec

    # Filter banks implementation
    # Generate filter banks and apply
    # Output: np.float32 array in [Batch X NData X NFrame]
    def apply_fbanks(self, data):

        # Convert Hz to Mel
        mel_low = 0
        mel_high = (2595 * np.log10(1 + (self.fs / 2) / 700))
        # Equally spaced in Mel scale
        mel_points = np.linspace(mel_low, mel_high, self.nfbanks + 2)
        # Convert Mel to Hz
        hz_points = (700 * (10**(mel_points / 2595) - 1))

        # Construct filter banks
        bins = np.floor((self.nfft + 1) * hz_points / self.fs)
        fbanks = np.zeros((self.nfbanks, int(np.floor(self.nfft / 2 + 1))))
        for m in range(1, self.nfbanks + 1):
            f_m_minus = int(bins[m - 1])
            f_m = int(bins[m])
            f_m_plus = int(bins[m + 1])

            for k in range(f_m_minus, f_m):
                fbanks[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
            for k in range(f_m, f_m_plus):
                fbanks[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

        # Apply filter
        trans_spec = np.transpose(data, axes=(0, 2, 1))
        filter_trans_spec = np.matmul(trans_spec, fbanks.T)
        filter_spec = np.transpose(filter_trans_spec, axes=(0, 2, 1))

        return filter_spec

    # Number to dB convertion with numerical stability check
    # Replace zeros with eps and convert to dB
    # Output: np.float32 array in [Batch X NData X NFrame]
    def num2dB(self, data):

        # Numerical Stability
        stab_data = np.where(data == 0,
                             np.finfo(self.out_dtype).eps,
                             data)

        # Convert to dB
        dB_data = 20 * np.log10(stab_data)

        return dB_data

    # Mean removal normalization
    # Remove mean alone axis default to 1
    # Output: np.float32 array in [Batch X NData X NFrame]
    def remove_mean(self, data, axis=1):

        # Calculate mean and parse dimension
        data_mean = np.expand_dims(np.mean(data, axis=axis), axis)

        # Remove mean
        ret = data - (data_mean + 1e-10)

        return ret


if __name__ == '__main__':

    print('Running feature.py as main file...')
