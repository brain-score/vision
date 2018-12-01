# -*- coding: utf-8 -*-

__author__ = """Jon Prescott-Roy"""
__email__ = 'jjpr@mit.edu'
__version__ = '0.1.0'

from .fetch import get_assembly, get_stimulus_set

from brainscore.contrib import benchmarks as contrib_benchmarks

contrib_benchmarks.inject()
