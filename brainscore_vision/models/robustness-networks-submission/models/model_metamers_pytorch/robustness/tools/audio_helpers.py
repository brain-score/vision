import scipy
import scipy.signal as signal
import scipy.io
import scipy.io.wavfile
import numpy as np

def load_audio_wav_resample(audio_path, DUR_SECS = 2, resample_SR = 16000, START_SECS=0, return_mono=True):
    """
    Loads a .wav file, chooses the length, and resamples to the desired rate.

    Parameters
    ----------
    audio_path : string
        path to the .wav file to load
    DUR_SECS : int/float, or 'full'
        length of the audio to load in in seconds, if 'full' loads the (remaining) clip
    resample_SR : float
        sampling rate for the output sound
    START_SECS : int/float, or 'random'
        where to start reading the sound, in seconds, unless 'random' to choose a random segment
    return_mono : Boolean
        if true, returns a mono version of the sound

    """
    SR, audio = scipy.io.wavfile.read(audio_path)
    if DUR_SECS!='full':
        if (len(audio))/SR<DUR_SECS:
            print("PROBLEM WITH LOAD AUDIO WAV: The sound is only %d second while you requested %d seconds long"%(int((len(audio))/SR), DUR_SECS))
            return
    if return_mono:
        if audio.ndim>1:
            audio = audio.sum(axis=1)/2
    if START_SECS=='random':
        start_idx = np.random.choice(len(audio)-int(DUR_SECS*SR))
        START_SECS = start_idx/SR
    if DUR_SECS!='full':
        audio = audio[int(START_SECS*SR):int(START_SECS*SR) + int(SR*DUR_SECS)]
    else:
        audio = audio[int(START_SECS*SR):]
    if SR != resample_SR:
        audio = signal.resample_poly(audio, resample_SR, SR, axis=0)
        SR = resample_SR
    return audio, SR
