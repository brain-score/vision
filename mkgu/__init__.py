# -*- coding: utf-8 -*-

__author__ = """Jon Prescott-Roy"""
__email__ = 'jjpr@mit.edu'
__version__ = '0.1.0'

from . import fetch


def get_assembly(name):
    return fetch.get_assembly(name)

