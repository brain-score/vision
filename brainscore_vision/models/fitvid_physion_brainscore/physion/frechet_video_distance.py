import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub

def fvd_preprocess(videos, target_resolution):
    videos = tf.convert_to_tensor(videos * 255.0, dtype=tf.float32)
    videos_shape = videos.shape.as_list()
    all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
    resized_videos = tf.image.resize(all_frames, size=target_resolution)
    target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
    output_videos = tf.reshape(resized_videos, target_shape)
    scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
    return scaled_videos

def create_id3_embedding(videos):
    """Get id3 embeddings."""
    module_spec = 'https://tfhub.dev/deepmind/i3d-kinetics-400/1'
    base_model = hub.load(module_spec)
    input_tensor = base_model.graph.get_tensor_by_name('input_frames:0')
    i3d_model = base_model.prune(input_tensor, 'RGB/inception_i3d/Mean:0')
    output = i3d_model(videos)
    return output

def calculate_fvd(real_activations, generated_activations):
    return tfgan.eval.frechet_classifier_distance_from_activations(
      real_activations, generated_activations)

def fvd(video_1, video_2):
    video_1 = fvd_preprocess(video_1, (224, 224))
    video_2 = fvd_preprocess(video_2, (224, 224))
    x = create_id3_embedding(video_1)
    y = create_id3_embedding(video_2)
    result = calculate_fvd(x, y)
    return result.numpy().tolist()
