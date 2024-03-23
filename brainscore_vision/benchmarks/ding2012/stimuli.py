import numpy as np

# - square with side of length 4 degrees (for monkey C, 3.5 for monkey F)
# - velocity 6 degrees per second
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to generate random dots
def generate_vel(speed, direction, dt):
    # Compute displacements based on speed and directions
    displacements = speed * dt
    dx = displacements * np.cos(np.radians(direction))
    dy = displacements * np.sin(np.radians(direction))

    return np.array((dx, dy))

"""
Because the dots stimulus is often used to understand decisions that involve integrating evidence over space and time, 
a lot of the weirdness of drawing the dots is to ensure that people can't use other strategies 
- which usually means avoiding having individual dots carry too much information. 
In other words, we want to avoid drawing it in a way that the subject can just track a single dot and figure out the answer.
The “thisFrame” variable is part of one of the main ways we do that. 
Basically instead of drawing a single set of dots, some fraction of which moves from frame-to-frame, 
we usually divide them up into three subsets. The “motion” is the displacement of dots from each subset every third frame.
"""

def make_dots(
        id, 
        density = 150, # 16.7,
        coherence = 0.03,  # Set coherence level (0 to 1)
        speed = 6.0,  # 6.0,  # Set motion speed
        direction = 90,  # Set motion direction (in degrees)
        fps = 60,
        screen_radius = 5.0,  # Radius of the circular display area
        screen_size = 8.0,
        dot_size = 10,
        duration = 2, # seconds
        replacement_frames = 3,
    ):

    dt = 1 / fps  # Time step (inverse of FPS)
    num_dots = int(density * screen_radius**2 / fps)
    
    dot_positions = np.random.uniform(-screen_radius, screen_radius, (2, num_dots))
    life_times = np.zeros(num_dots, dtype=int)

    # Create the animation
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    # Function to update dot positions in the animation
    def update(frame):
        ax.clear()
        this_frame = np.zeros(num_dots, dtype=bool)
        this_frame[frame % replacement_frames::replacement_frames] = True

        # prioritize short life times
        num_consistent_dots = int(coherence * num_dots)
        consistent_dots_ = np.argsort(life_times)[:num_consistent_dots]
        consistent_dots = np.zeros(num_dots, dtype=bool)
        consistent_dots[consistent_dots_] = True
        non_consistent_dots = ~consistent_dots
        consistent_dots[~this_frame] = False
        non_consistent_dots[~this_frame] = False

        life_times[consistent_dots] += 1

        dot_positions[:, consistent_dots] += generate_vel(speed, direction, dt)[:,None] * replacement_frames
        dot_positions[:, non_consistent_dots] = np.random.uniform(-screen_radius, screen_radius, (2, non_consistent_dots.sum()))
        # dot_positions[:, non_consistent_dots] += generate_vel(speed, np.random.uniform(0, 360, non_consistent_dots.sum()), dt) * replacement_frames

        # wrap around
        dot_positions[0, :] = np.mod(dot_positions[0, :] + screen_radius, 2*screen_radius) - screen_radius
        dot_positions[1, :] = np.mod(dot_positions[1, :] + screen_radius, 2*screen_radius) - screen_radius

        # Only display dots within the circular area
        mask = np.sqrt(dot_positions[0, :]**2 + dot_positions[1, :]**2) < screen_radius
        ax.set_xlim(-screen_size, screen_size)
        ax.set_ylim(-screen_size, screen_size)  
        # axis off
        ax.axis('off')
        # Plot the dots
        ax.scatter(dot_positions[0, mask], dot_positions[1, mask], s=dot_size, c='white')

    ani = FuncAnimation(fig, update, frames=fps * duration, interval=1000 / fps)
    # save mp4
    ani.save(f'videos/{id}.mp4', writer='ffmpeg', fps=fps)

if __name__ == "__main__":
    from joblib import Parallel, delayed

    coherences = [0.032, 0.064, 0.128, 0.256, 0.512]
    # directions = [0, 45, 90, 135, 180, 225, 270, 315]
    directions = [0, 180]

    # # test
    # make_dots("test1", coherence=0.032, direction=0)
    # make_dots("test2", coherence=0.032, direction=180)

    # make_dots("test3", coherence=0.128, direction=0)
    # make_dots("test4", coherence=0.128, direction=180)
    
    # # make 50 training samples for each case
    # for i in range(50):
    #     for coh in coherences:
    #         for dir in directions:
    #             make_dots(f"train_{coh}_{dir}_{i}", coherence=coh, direction=dir)

    # parallize
    Parallel(n_jobs=-1)(delayed(make_dots)(f"train_{coh*100}_{dir}_{i}", coherence=coh, direction=dir) 
                        for i in range(100) for coh in coherences for dir in directions)

    # make test samples:
    # for coh in coherences:
    #     for dir in directions:
    #         make_dots(f"test_{coh}_{dir}", cohercoh, direction=dir)

    # parallize
    Parallel(n_jobs=-1)(delayed(make_dots)(f"test_{coh*100}_{dir}_{i}", coherence=coh, direction=dir) 
                        for i in range(100) for coh in coherences for dir in directions)

    # make_dots("test1", coherence=.512, direction=180)
    # make_dots("test2", coherence=0.256, direction=0)
    # make_dots("test3", coherence=0.032, direction=180)