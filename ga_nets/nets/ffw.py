#!/usr/bin/env python3
#
# @Author(s):
#    - Tomas Bartoli <tomasbartoli1992@gmail.com>
# @Date: 09/06/2020 (dd/mm/yyyy)
# @since: 1.0.0
#
#    Copyright (C) 2020  Tomas Bartoli
#
#    This isn't a free software, if you steal it... then, good for you.
"""Feedforward Neural Network"""
# pylint: disable=too-few-public-methods
from ga_nets.connection import SynapseDirection
from ga_nets.neuron import Neuron
from ga_nets.network import Network


class FeedForward(Network):
    """Classico wrapper per il feedforward network"""
    def __init__(self, traits):
        super().__init__(traits, FFWNeuron)


class FFWNeuron(Neuron):
    """Wrapper per i neuroni nei feedforward networks"""
    def activate(self):
        """Please see: @fsneat.architecture.network.Network.activate()"""
        states = [s.from_neuron.state[0] * s.weight
                  for s in self.synapses[SynapseDirection.IN.value]
                  if s.from_neuron.state]

        # Per evitare errori dati dal min() e max()
        if not states:
            states.append(0)

        self.state.append(self.squash(self.aggregation(states) + self.bias))
