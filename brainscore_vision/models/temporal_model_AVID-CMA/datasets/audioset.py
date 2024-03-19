# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import csv
import numpy as np
import glob
from .video_db import VideoDataset

DATA_PATH = '/data/datasets/AS240/data/'
CACHE_PATH = 'datasets/cache/audioset'


class AudiosetClasses:
    def __init__(self):
        ann_list = list(csv.DictReader(open(CACHE_PATH + '/class_labels_indices.csv')))
        self.classes = [ann['mid'] for ann in ann_list]
        self.class_label = {ann['mid']: int(ann['index']) for ann in ann_list}
        self.display_name = {ann['mid']: ann['display_name'] for ann in ann_list}

    def __getitem__(self, index):
        return self.display_name[self.classes[index]]

    def __len__(self):
        return len(self.classes)

    def class2index(self, class_string):
        return self.class_label[class_string]


class AudioSet(VideoDataset):
    def __init__(self, subset,
                 return_video=True,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 return_audio=False,
                 audio_clip_duration=1.,
                 audio_fps=None,
                 audio_fps_out=64,
                 audio_transform=None,
                 return_labels=False,
                 return_index=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=1,
                 ):

        root = f"{DATA_PATH}/{subset.split('-')[0]}_segments/video"

        classes = AudiosetClasses()
        filenames = [f"{ln.strip().split()[0]}" for ln in open(f"{CACHE_PATH}/{subset}.txt")]
        available = set([fn.split('/')[-1].split('.')[0] for fn in glob.glob(f"{root}/*")])
        filenames = [fn for fn in filenames if fn.split('.')[0] in available]

        assert return_labels is False
        labels = None

        super(AudioSet, self).__init__(
            return_video=return_video,
            video_clip_duration=video_clip_duration,
            video_root=root,
            video_fns=filenames,
            video_fps=video_fps,
            video_transform=video_transform,
            return_audio=return_audio,
            audio_clip_duration=audio_clip_duration,
            audio_root=root,
            audio_fns=filenames,
            audio_fps=audio_fps,
            audio_fps_out=audio_fps_out,
            audio_transform=audio_transform,
            return_labels=return_labels,
            labels=labels,
            return_index=return_index,
            max_offsync_augm=max_offsync_augm,
            mode=mode,
            clips_per_video=clips_per_video,
        )

        self.name = 'AudioSet dataset'
        self.root = root
        self.subset = subset

        self.num_videos = len(filenames)
        self.num_classes = len(classes)

        self.sample_id = np.array([fn.split('.')[0].encode('utf-8') for fn in filenames])
