import pytest
import brainscore

def test_gallant():
    gallant = brainscore.get_assembly("gallant.David2004")

def test_hvm():
    hvm = brainscore.get_assembly("dicarlo.Majaj2015")

def test_hvm_temporal():
    hvm_t = brainscore.get_assembly("dicarlo.Majaj2015.temporal")

def test_tolias():
    tolias = brainscore.get_assembly("tolias.Cadena2017")
