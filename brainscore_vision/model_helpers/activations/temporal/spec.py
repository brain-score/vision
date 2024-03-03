from copy import deepcopy


class Spec(dict):
    """Specification for the model. It is a dictionary with the following keys:

    - input: dict
        Specification for the input. It can be the following:

        * type = image

            size: tuple, optional
                The size of the image.
            channels: str, optional
                The channels of the image, e.g. RGB, BGR, etc.
        
        * type = video

            fps: int, optional
                The frames per second of the video.
            duration: float, optional
                The duration of the video in seconds.
            num_frames: int, optional
                The number of frames in the video.
                
    - activation: dict
        Specification for the activation. It has the following format: {layer_name: activation_dims}
        where activation_dims is an iterable of channel names. The channel names can be the following:
            * C: channel
            * T: time
            * H: height
            * W: width
            * K: token

    - model: dict, Optional
        Specification for the model. It can include the following:

        objective: str, optional
            The objective of the model.
        dataset: str, optional
            The dataset the model was trained on.
        architecture: choose from this module, optional
            The architecture of the model.
        bib: str, optional
            The bibtex of the paper the model was introduced in.   
        source: str, optional
            The source where the model was implemented.
        stats: float, optional
            The statistics of the model, including task performance, number of parameters, etc.
        ...
    """

    def _check(self):
        assert 'input' in self, "input spec is required"
        assert 'type' in self['input'], "input type is missing"
        assert 'activation' in self, "activation spec is required"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._check()

    def copy(self):
        return deepcopy(self)

    def extend(self, other):
        new_spec = self.copy()
        new_spec.update(other)
        return new_spec


class add_spec:
    def __init__(self, spec):
        if isinstance(spec, dict):
            spec = Spec(spec)
        self.spec = spec

    def __call__(self, cls):
        cls.spec = self.spec.extend(cls.spec)
        return cls