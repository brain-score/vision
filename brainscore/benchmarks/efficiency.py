from brainscore.metrics import Score

from brainscore.benchmarks import Benchmark


class NeuronsSynapsesEnergy(Benchmark):
    E_mac = 3.2e-12
    E_mem = 5e-12

    @property
    def identifier(self):
        return 'roy.Chakraborty2019-neurons_synapses'

    def __call__(self, candidate):
        weights = candidate.synapses
        neurons = candidate.neurons
        num_weights, num_neurons = len(weights), len(neurons)
        energy = self.E_mac * num_weights + self.E_mem * num_neurons
        return Score(energy, coords={'unit': 'mJ'})
