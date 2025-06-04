import torch
from spikerplus import NetBuilder

def build_network(cfg):
    example_input = torch.zeros(cfg.max_len, cfg.n_freq_bins)
    net_dict = {
        "n_cycles": example_input.shape[0],
        "n_inputs": example_input.shape[1],
        "layer_0": {"neuron_model": "syn", "n_neurons": 256, "alpha": 0.05, "beta": 0.05, "threshold": 0.8},
        "layer_1": {"neuron_model": "rsyn", "n_neurons": 128, "alpha": 0.07, "beta": 0.07, "threshold": 0.3},
        "layer_2": {"neuron_model": "rif", "n_neurons": 64, "beta": 0.4, "threshold": 0.25},
        "layer_3": {"neuron_model": "syn", "n_neurons": 128, "alpha": 0.1, "beta": 0.6, "threshold": 0.2},
        "layer_4": {"neuron_model": "if", "n_neurons": cfg.n_freq_bins, "threshold": 0.06},
    }
    return NetBuilder(net_dict).build()
