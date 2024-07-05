from brainscore_vision import benchmark_registry
from . import benchmark

benchmark_registry['Malania2007.short2'] = lambda: benchmark._Malania2007Base('short-2')
benchmark_registry['Malania2007.short4'] = lambda: benchmark._Malania2007Base('short-4')
benchmark_registry['Malania2007.short6'] = lambda: benchmark._Malania2007Base('short-6')
benchmark_registry['Malania2007.short8'] = lambda: benchmark._Malania2007Base('short-8')
benchmark_registry['Malania2007.short16'] = lambda: benchmark._Malania2007Base('short-16')
benchmark_registry['Malania2007.equal2'] = lambda: benchmark._Malania2007Base('equal-2')
benchmark_registry['Malania2007.long2'] = lambda: benchmark._Malania2007Base('long-2')
benchmark_registry['Malania2007.equal16'] = lambda: benchmark._Malania2007Base('equal-16')
benchmark_registry['Malania2007.long16'] = lambda: benchmark._Malania2007Base('long-16')
benchmark_registry['Malania2007.vernieracuity'] = lambda: benchmark._Malania2007VernierAcuity()
