from .video import Video

input_type_mapping = {
    "video": Video
}

def get_input_cls(input_type):
    return input_type_mapping[input_type]