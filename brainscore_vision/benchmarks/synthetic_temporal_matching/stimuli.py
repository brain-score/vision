"""
Synthetic moving dot stimulus generation for temporal match-to-sample benchmarks.

Generates simple video stimuli with dots moving in different directions.
The matching feature is motion direction.
"""

import numpy as np
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

# Optional: try to import video writing library
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def generate_moving_dot_frames(
    direction: float,
    n_frames: int = 16,
    frame_size: Tuple[int, int] = (64, 64),
    dot_radius: int = 4,
    speed: float = 2.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate frames of a dot moving in a specified direction.

    :param direction: Motion direction in radians (0 = right, pi/2 = up)
    :param n_frames: Number of frames to generate
    :param frame_size: (height, width) of each frame
    :param dot_radius: Radius of the dot in pixels
    :param speed: Pixels per frame
    :param seed: Random seed for initial position
    :return: Array of shape (n_frames, height, width, 3) with uint8 values
    """
    if seed is not None:
        np.random.seed(seed)

    height, width = frame_size
    frames = np.zeros((n_frames, height, width, 3), dtype=np.uint8)

    # Start position (center with some randomness)
    start_x = width // 2 + np.random.randint(-10, 10)
    start_y = height // 2 + np.random.randint(-10, 10)

    # Velocity components
    vx = speed * np.cos(direction)
    vy = -speed * np.sin(direction)  # Negative because y increases downward

    for i in range(n_frames):
        # Current position
        x = int(start_x + vx * i) % width
        y = int(start_y + vy * i) % height

        # Draw dot (simple circle)
        for dx in range(-dot_radius, dot_radius + 1):
            for dy in range(-dot_radius, dot_radius + 1):
                if dx * dx + dy * dy <= dot_radius * dot_radius:
                    px = (x + dx) % width
                    py = (y + dy) % height
                    frames[i, py, px] = [255, 255, 255]  # White dot on black

    return frames


def save_frames_as_video(
    frames: np.ndarray,
    output_path: str,
    fps: int = 8
) -> str:
    """
    Save frames as a video file.

    :param frames: Array of shape (n_frames, height, width, 3)
    :param output_path: Path to save the video
    :param fps: Frames per second
    :return: Path to the saved video
    """
    if not HAS_CV2:
        raise ImportError("cv2 (opencv-python) is required to save videos")

    n_frames, height, width, _ = frames.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()
    return output_path


def generate_trial_stimuli(
    trial_id: str,
    sample_direction: float,
    choice_directions: List[float],
    correct_choice_index: int,
    output_dir: str,
    n_frames: int = 16,
    frame_size: Tuple[int, int] = (64, 64),
    seed: Optional[int] = None
) -> dict:
    """
    Generate stimuli for a single match-to-sample trial.

    :param trial_id: Identifier for this trial
    :param sample_direction: Direction for the sample stimulus (radians)
    :param choice_directions: Directions for each choice stimulus
    :param correct_choice_index: Which choice matches the sample
    :param output_dir: Directory to save video files
    :param n_frames: Frames per video
    :param frame_size: Frame dimensions
    :param seed: Random seed base
    :return: Dict with stimulus metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    stimuli = []
    base_seed = seed if seed is not None else hash(trial_id) % (2**31)

    # Generate sample stimulus
    sample_id = f"{trial_id}_sample"
    sample_frames = generate_moving_dot_frames(
        direction=sample_direction,
        n_frames=n_frames,
        frame_size=frame_size,
        seed=base_seed
    )
    sample_path = os.path.join(output_dir, f"{sample_id}.mp4")

    if HAS_CV2:
        save_frames_as_video(sample_frames, sample_path)
    else:
        # Save as numpy for testing without cv2
        np.save(sample_path.replace('.mp4', '.npy'), sample_frames)
        sample_path = sample_path.replace('.mp4', '.npy')

    stimuli.append({
        'stimulus_id': sample_id,
        'trial_id': trial_id,
        'stimulus_role': 'sample',
        'choice_index': np.nan,
        'direction': sample_direction,
        'stimulus_path': sample_path,
    })

    # Generate choice stimuli
    for i, direction in enumerate(choice_directions):
        choice_id = f"{trial_id}_choice_{i}"
        choice_frames = generate_moving_dot_frames(
            direction=direction,
            n_frames=n_frames,
            frame_size=frame_size,
            seed=base_seed + i + 1
        )
        choice_path = os.path.join(output_dir, f"{choice_id}.mp4")

        if HAS_CV2:
            save_frames_as_video(choice_frames, choice_path)
        else:
            np.save(choice_path.replace('.mp4', '.npy'), choice_frames)
            choice_path = choice_path.replace('.mp4', '.npy')

        stimuli.append({
            'stimulus_id': choice_id,
            'trial_id': trial_id,
            'stimulus_role': 'choice',
            'choice_index': i,
            'direction': direction,
            'stimulus_path': choice_path,
        })

    return {
        'stimuli': stimuli,
        'correct_choice': correct_choice_index,
        'sample_direction': sample_direction,
    }


def generate_synthetic_benchmark_stimuli(
    n_trials: int = 10,
    n_choices: int = 3,
    output_dir: Optional[str] = None,
    seed: int = 42
) -> Tuple[List[dict], List[int]]:
    """
    Generate a complete set of synthetic stimuli for the benchmark.

    :param n_trials: Number of trials to generate
    :param n_choices: Number of choices per trial
    :param output_dir: Directory to save stimuli (uses temp dir if None)
    :param seed: Random seed for reproducibility
    :return: (list of stimulus dicts, list of correct choice indices)
    """
    np.random.seed(seed)

    if output_dir is None:
        output_dir = os.path.join(tempfile.gettempdir(), 'synthetic_temporal_matching')

    os.makedirs(output_dir, exist_ok=True)

    all_stimuli = []
    correct_choices = []

    # Define possible directions (8 cardinal/ordinal directions)
    directions = [i * np.pi / 4 for i in range(8)]

    for trial_idx in range(n_trials):
        trial_id = f"trial_{trial_idx:03d}"

        # Pick sample direction
        sample_direction = directions[np.random.randint(len(directions))]

        # Pick choice directions - one matches sample, others are different
        correct_idx = np.random.randint(n_choices)
        choice_directions = []

        for i in range(n_choices):
            if i == correct_idx:
                choice_directions.append(sample_direction)
            else:
                # Pick a different direction
                other_dirs = [d for d in directions if d != sample_direction]
                choice_directions.append(other_dirs[np.random.randint(len(other_dirs))])

        trial_data = generate_trial_stimuli(
            trial_id=trial_id,
            sample_direction=sample_direction,
            choice_directions=choice_directions,
            correct_choice_index=correct_idx,
            output_dir=output_dir,
            seed=seed + trial_idx
        )

        all_stimuli.extend(trial_data['stimuli'])
        correct_choices.append(trial_data['correct_choice'])

    return all_stimuli, correct_choices
