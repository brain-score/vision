from typing import Union, List
from pathlib import Path

from brainscore_vision.model_helpers.activations.temporal.inputs import Video, Stimulus
from ..base import TemporalInferencer


MS_ROUNDING_DIGITS = 3

class TemporalContextInferencerBase(TemporalInferencer):
    """Inferencer base that computes the temporal context for concrete context-based inferencers,
    like CausalInferencer and BlockInferencer. 

    The context range is first determined by the num_frames and duration. Then, temporal_context_strategy
    is used to determine the lower bound of temporal context and the expected temporal context.
    
    Parameters
    ----------
    temporal_context_strategy: str
        specify how the length of temporal context for causal inference is determined.
        Options:
        - "greedy": the length of the temporal context is determined by the maximum of num_frames and duration.
        - "conservative": the length of the temporal context is determined by the minimum of num_frames and duration.
        - "fix": the length of the temporal context is determined by the specified "fixed_temporal_context".
    
    fixed_temporal_context: float
        specify the fixed length of the temporal context, in ms. It will be used only if temporal_context_strategy is "fix".
    
    out_of_bound_strategy: str
        specify how to handle the out-of-bound temporal context.
        Options:
        - "repeat": the out-of-bound temporal context will be repeated.
        - TODO: "black": the out-of-bound temporal context will be zero-padded.
        - TODO: "gray": the out-of-bound temporal context will be 128-padded.
    """
    def __init__(
            self, 
            *args,
            temporal_context_strategy : str = "greedy",
            fixed_temporal_context : float = None,
            out_of_bound_strategy : str = "repeat",
            **kwargs
        ):
        self.temporal_context_strategy = temporal_context_strategy
        self.fixed_temporal_context = fixed_temporal_context
        self.out_of_bound_strategy = out_of_bound_strategy
        if self.temporal_context_strategy == "fix" and self.fixed_temporal_context is None:
            raise ValueError("fixed_temporal_context must be specified if temporal_context_strategy is 'fix'.")
        super().__init__(*args, **kwargs)

    @property
    def identifier(self) -> str:
        lower, context = self._compute_temporal_context()
        lower = round(lower, MS_ROUNDING_DIGITS)
        context = round(context, MS_ROUNDING_DIGITS)
        to_add = f".context={context}>{lower}.oob={self.out_of_bound_strategy}"
        return f"{super().identifier}{to_add}"
        
    def load_stimulus(self, path: Union[str, Path]) -> Video:
        if self.convert_to_video and Stimulus.is_image_path(path):
            video = Video.from_img_path(path, self.img_duration, self.fps)
        else:
            video = Video.from_path(path)
        video = video.set_fps(self.fps)
        # does no check here
        return video

    def _overlapped_range(self, start1, end1, start2, end2):
        # compute the overlapped range of two ranges (start1, end1) and (start2, end2)
        lower, upper = max(start1, start2), min(end1, end2)
        if lower > upper:
            raise ValueError(f"Ranges [{start1}, {end1}] and [{start2}, {end2}] do not overlap.")
        return lower, upper
        
    def _compute_temporal_context(self):
        duration = self.duration
        num_frames = self.num_frames
        strategy = self.temporal_context_strategy

        interval = 1000 / self.fps
        num_frames_implied_ran = (num_frames[0] * interval, num_frames[1] * interval)
        ran = self._overlapped_range(*num_frames_implied_ran, *duration)
        lower = ran[0]

        if strategy in ["greedy", "conservative"]:
            if strategy == "greedy":
                return lower, ran[1]
            elif strategy == "conservative":
                return lower, ran[0]

        elif strategy == "fix":
            context = self.fixed_temporal_context
            assert ran[0] <= context <= ran[1], f"Fixed temporal context {context} is not within the range {ran}"

        else:
            raise ValueError(f"Unknown temporal context strategy: {strategy}")

        return lower, context
    
