# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import av
import numpy as np
from fractions import Fraction
av.logging.set_level(0)

def av_open(inpt):
    return av.open(inpt)


def av_load_video(container, video_fps=None, start_time=0, duration=None):
    video_stream = container.streams.video[0]
    _ss = video_stream.start_time * video_stream.time_base
    _dur = video_stream.duration * video_stream.time_base
    _ff = _ss + _dur
    _fps = video_stream.average_rate

    if video_fps is None:
        video_fps = _fps

    if duration is None:
        duration = _ff - start_time

    # Figure out which frames to decode
    outp_times = [t for t in np.arange(start_time, min(start_time + duration - 0.5/_fps, _ff), 1./video_fps)][:int(duration*video_fps)]
    outp_vframes = [int((t - _ss) * _fps) for t in outp_times]
    start_time = outp_vframes[0] / float(_fps)

    # Fast forward
    container.seek(int(start_time * av.time_base))

    # Decode snippet
    frames = []
    for frame in container.decode(video=0):
        if len(frames) == len(outp_vframes):
            break   # All frames have been decoded
        frame_no = frame.pts * frame.time_base * _fps
        if frame_no < outp_vframes[len(frames)]:
            continue    # Not the frame we want

        # Decode
        pil_img = frame.to_image()
        while frame_no >= outp_vframes[len(frames)]:
            frames += [pil_img]
            if len(frames) == len(outp_vframes):
                break   # All frames have been decoded

    return frames, video_fps, start_time


def av_laod_audio(container, audio_fps=None, start_time=0, duration=None):
    audio_stream = container.streams.audio[0]
    _ss = audio_stream.start_time * audio_stream.time_base if audio_stream.start_time is not None else 0.
    _dur = audio_stream.duration * audio_stream.time_base
    _ff = _ss + _dur
    _fps = audio_stream.rate

    if audio_fps is None:
        resample = False
        audio_fps = _fps
    else:
        resample = True
        audio_resampler = av.audio.resampler.AudioResampler(format="s16p", layout="mono", rate=audio_fps)

    if duration is None:
        duration = _ff - start_time
    duration = min(duration, _ff - start_time)
    end_time = start_time + duration

    # Fast forward
    container.seek(int(start_time * av.time_base))

    # Decode snippet
    data, timestamps = [], []
    for frame in container.decode(audio=0):
        frame_pts = frame.pts * frame.time_base
        frame_end_pts = frame_pts + Fraction(frame.samples, frame.rate)
        if frame_end_pts < start_time:   # Skip until start time
            continue
        if frame_pts > end_time:       # Exit if clip has been extracted
            break

        try:
            frame.pts = None
            if resample:
                np_snd = audio_resampler.resample(frame).to_ndarray()
            else:
                np_snd = frame.to_ndarray()
            data += [np_snd]
            timestamps += [frame_pts]
        except AttributeError:
            break
    data = np.concatenate(data, 1)

    # Trim audio
    start_decoded_time = timestamps[0]
    ss = int((start_time - start_decoded_time) * audio_fps)
    t = int(duration * audio_fps)
    if ss < 0:
        data = np.pad(data, ((0, 0), (-ss, 0)), 'constant', constant_values=0)
        ss = 0
    if t > data.shape[1]:
        data = np.pad(data, ((0, 0), (0, t-data.shape[1])), 'constant', constant_values=0)
    data = data[:, ss: ss+t]
    data = data / np.iinfo(data.dtype).max

    return data, audio_fps


