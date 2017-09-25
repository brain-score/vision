from __future__ import absolute_import, division, print_function, unicode_literals


class DataPoint(object):
    """A DataPoint represents one value, usually a recording from one neuron or node,
    in response to one presentation of a stimulus.  """
    def __init__(self, value, neuroid, presentation):
        self.value = value
        self.neuroid = neuroid
        self.presentation = presentation


class Presentation(object):
    """An instance of presenting a stimulus to a system, to evoke a response.  """
    def __init__(self, stimulus):
        self.stimulus = stimulus


class Stimulus(object):
    """The stimulus which evokes a recorded response.  """
    def __init__(self):
        pass


class Neuroid(object):
    """The neuron or neuron-like thing from which data in a DataPoint is recorded.  """
    def __init__(self, subject):
        self.subject = subject


class Subject(object):
    """The system, biological or artificial, which contains neuroids under study.  """
    def __init__(self, name, type):
        self.name = name
        self.type = type

