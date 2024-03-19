# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from .video_db import VideoDataset

DATA_PATH = '/data/datasets/UCF101/data'
ANNO_PATH = '/data/datasets/UCF101/ucfTrainTestlist/'


class UCF(VideoDataset):
    def __init__(self, subset,
                 video_clip_duration=0.5,
                 return_video=True,
                 video_fps=16.,
                 video_transform=None,
                 return_audio=False,
                 return_labels=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=20,
                 ):

        assert return_audio is False
        self.name = 'UCF-101'
        self.root = DATA_PATH
        self.subset = subset

        classes_fn = f'{ANNO_PATH}/classInd.txt'
        self.classes = [l.strip().split()[1] for l in open(classes_fn)]

        filenames = [ln.strip().split()[0] for ln in open(f'{ANNO_PATH}/{subset}.txt')]
        labels = [fn.split('/')[0] for fn in filenames]
        labels = [self.classes.index(cls) for cls in labels]

        self.num_classes = len(self.classes)
        self.num_videos = len(filenames)

        super(UCF, self).__init__(
            return_video=return_video,
            video_root=DATA_PATH,
            video_clip_duration=video_clip_duration,
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

