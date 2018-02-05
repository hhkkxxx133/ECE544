# config.py
# Created by dwang49
# This files serves as the header file of the project
# Here defines the configuration of the multiple blocks being used


import numpy as np


class workspace_config():

    # Path to workspace
    root_path = '/data/khui3/tfaudio/'

    # Path to code
    code_path = root_path + 'code/'

    # Path to training data
    train_path = root_path + 'train/'

    # Path to validation data
    val_path = root_path + 'val/'

    # Path to test data
    test_path = root_path + 'test/'


# Configuration for RNN model
class model_config():

    # LSTM cell type
    lstm_cell_type = 'BasicLSTMCell'

    # Hidden units number
    hidden_units_num = 128

    # Save path
    save_path = './saved_models/'

    # Save model number
    save_model_num = 5


# Configuration for training
class train_config():

    # Batch Size
    batch_size = 40

    # Learning rate
    learning_rate = 0.0001

    # Number of steps
    num_steps = 500000

    # Cuda device
    cuda_device = '/cpu:0'


# Configuration of audio feature extraction
class feature_config():

    # Feature extraction method, valid methods includes:
    # 'time': use time domain samples
    # 'stft': use full spectrogram from short time fourier transform
    # 'fbank': use filter banks
    # 'mfcc': use mel-frequency cepstral coefficients
    feature_method = 'fbank'

    # Audio data sampling rate in Hz
    fs = 16000

    # Pre-Emphasis: set to 0 to avoid preemphasis, otherwise apply 0.95 or 0.97
    pre_emphasis_coef = 0.97

    # Feature frame length in milliseconds
    frame_length = 25

    # Overlapping length between frames in milliseconds
    overlap_length = 15

    # FFT length
    nfft = 512

    # Filter bank number
    nfbanks = 40

    # Mel-frequency cepstral coeffs number
    nmfccs = 12

    # Mean Removal Normalization
    norm_mean = True

    # Output Data Type
    out_dtype = np.float32
