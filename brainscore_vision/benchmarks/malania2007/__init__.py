from brainscore_vision import benchmark_registry
from . import benchmark

benchmark_registry['Malania2007.short2-threshold_elevation'] = lambda: benchmark._Malania2007Base('short2')
benchmark_registry['Malania2007.short4-threshold_elevation'] = lambda: benchmark._Malania2007Base('short4')
benchmark_registry['Malania2007.short6-threshold_elevation'] = lambda: benchmark._Malania2007Base('short6')
benchmark_registry['Malania2007.short8-threshold_elevation'] = lambda: benchmark._Malania2007Base('short8')
benchmark_registry['Malania2007.short16-threshold_elevation'] = lambda: benchmark._Malania2007Base('short16')
benchmark_registry['Malania2007.equal2-threshold_elevation'] = lambda: benchmark._Malania2007Base('equal2')
benchmark_registry['Malania2007.long2-threshold_elevation'] = lambda: benchmark._Malania2007Base('long2')
benchmark_registry['Malania2007.equal16-threshold_elevation'] = lambda: benchmark._Malania2007Base('equal16')
benchmark_registry['Malania2007.long16-threshold_elevation'] = lambda: benchmark._Malania2007Base('long16')
benchmark_registry['Malania2007.vernieracuity-threshold'] = lambda: benchmark._Malania2007VernierAcuity()
