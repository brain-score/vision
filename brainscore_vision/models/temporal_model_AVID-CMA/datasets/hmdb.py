# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from .video_db import VideoDataset

DATA_PATH = '/data/datasets/hmdb/videos'
ANNO_PATH = '/data/datasets/hmdb/splits/'


class HMDB(VideoDataset):
    def __init__(self, subset,
                 return_video=True,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 return_audio=False,
                 return_labels=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=20,
                 ):
        assert return_audio is False
        self.name = 'HMDB-101'
        self.root = DATA_PATH
        self.subset = subset

        # Get filenames
        classes = sorted(os.listdir(DATA_PATH))
        subset, split = subset.split('-')
        subset_id = {'train': '1', 'test': '2'}[subset]
        filenames, labels = [], []
        for cls in classes:
            for ln in open(f'{ANNO_PATH}/{cls}_test_{split}.txt'):
                fn, ss = ln.strip().split()
                if ss == subset_id:
                    filenames += [f"{cls}/{fn}"]
                    labels += [classes.index(cls)]

        self.classes = classes
        self.num_classes = len(self.classes)
        self.num_videos = len(filenames)

        super(HMDB, self).__init__(
            return_video=return_video,
            video_clip_duration=video_clip_duration,
            video_root=DATA_PATH,
            video_fns=filenames,
            video_fps=video_fps,
            video_transform=video_transform,
            return_audio=False,
            return_labels=return_labels,
            labels=labels,
            max_offsync_augm=max_offsync_augm,
            mode=mode,
            clips_per_video=clips_per_video,
        )
