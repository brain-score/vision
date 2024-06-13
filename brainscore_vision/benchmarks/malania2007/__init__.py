from brainscore_vision import benchmark_registry
from . import benchmark

benchmark_registry['Malania2007_short-2'] = lambda: benchmark._Malania2007Base('short-2')
benchmark_registry['Malania2007_short-4'] = lambda: benchmark._Malania2007Base('short-4')
benchmark_registry['Malania2007_short-6'] = lambda: benchmark._Malania2007Base('short-6')
benchmark_registry['Malania2007_short-8'] = lambda: benchmark._Malania2007Base('short-8')
benchmark_registry['Malania2007_short-16'] = lambda: benchmark._Malania2007Base('short-16')
benchmark_registry['Malania2007_equal-2'] = lambda: benchmark._Malania2007Base('equal-2')
benchmark_registry['Malania2007_long-2'] = lambda: benchmark._Malania2007Base('long-2')
benchmark_registry['Malania2007_equal-16'] = lambda: benchmark._Malania2007Base('equal-16')
benchmark_registry['Malania2007_long-16'] = lambda: benchmark._Malania2007Base('long-16')
