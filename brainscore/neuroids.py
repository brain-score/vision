from __future__ import absolute_import, division, print_function, unicode_literals


class Subject(object):
    """The system, biological or artificial, which contains neuroids under study.  """
    def __init__(self, name, type):
        self.name = name
        self.type = type


class Neuroid(object):
    """The neuron or neuron-like thing from which data in a DataPoint is recorded.  """
    def __init__(self, subject):
        self.subject = subject


