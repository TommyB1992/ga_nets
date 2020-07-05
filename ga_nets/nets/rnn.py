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
"""Recurrent Neural Network"""
from ga_nets.connection import (AlreadyConn, GateDirection, is_connected,
                                Synapse, SynapseDirection)
from ga_nets.neuron import Neuron
from ga_nets.network import Network


class Recurrent(Network):
    """Wrapper per il recurrent network"""
    def __init__(self, traits):
        super().__init__(traits, RNNNeuron)

        # L'indice è il neurone di partenza e la chiave è una lista con i
        # neuroni d'uscita
        # self.__gates = {}
        self.__gates = []

    def __str__(self):
        """Aggiunge le informazioni sui gates"""
        output = super().__str__()

        # if self.gates:
        #    output += "   Gates:\n"
        #    for from_neuron, to_neuron in self.gates.items():
        #        output += "      From {} to:\n".format(from_neuron.key)
        #        for neuron in to_neuron:
        #            output += "         -> {} (w: {})\n".format(
        #                neuron.key, from_neuron.gates[neuron])

        if self.gates:
            output += "  Gates:\n"
            for gate in self.gates:
                output += "      {}\n".format(gate)

        return output

    @property
    def gates(self):
        """Getter dei gates della connessione.

        Returns:
          La lista con i gates.
        """
        return self.__gates

    @gates.setter
    def gates(self, gates):
        self.__gates = gates

    def add_gate(self, from_neuron, to_neuron, weight=None):
        """Crea il gate fra neuroni.

        Args:
          from_neuron: Istanza del neurone di partenza.
          to_neuron: Istanza del neurone d'arrivo dal quale calcolare lo
                     stato precedente.
          weight: Peso del gate, se non definita viene assegnata tramite la
                  funzione assegnata nei 'traits'.

        Returns:
          L'istanza del gate creato.

        Raises:
          AlreadyConn: Se la connessione è già presente.
        """
        if is_connected(self.gates, from_neuron, to_neuron):
            AlreadyConn("Gate already present.")

        if weight is None:
            weight = self.traits["weight_fn"](self)

        gate = from_neuron.add_gate(to_neuron, weight)
        self.gates.append(gate)

        return gate

    def sub_gate(self, gate):
        """Rimuove il gate fra i due neuroni.

        Args:
          gate: Istanza del gate da rimuovere.
        """
        self.gate.remove(gate)
        gate.remove()

    def sub_neuron(self, neuron):
        """Rimuove il neurone dal network e in aggiunta anche i gates.

        Args:
          neuron: Istanza del neurone da rimuovere.
        """
        super().remove_neuron(neuron)

        # Rimuove i gates
        for direction in GateDirection:
            for gate in neuron.gates[direction.value]:
                self.gates.remove(gate)


class RNNNeuron(Neuron):
    """Wrapper per i neuroni nei recurrent networks"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.__gates = {}
        self.__gates = [[] for _ in GateDirection]

    @property
    def gates(self):
        """Getter dei gates della connessione.

        Returns:
          La lista con i gates.
        """
        return self.__gates

    @gates.setter
    def gates(self, gates):
        self.__gates = gates

    def activate(self):
        """Please see: @fsneat.architecture.network.Network.activate()"""
        step = len(self.state) - 1

        states = [s.from_neuron.state[-1] * s.weight
                  for s in self.synapses[SynapseDirection.IN.value]
                  if s.from_neuron.state]

        # Calcola gli stati precedenti
        # prev_states = (sum([g.state[step] * w
        #                    for g, w in self.gates.items()
        #                    if g.state])
        #               if step > -1 else 0)
        prev_states = (sum([g.to_neuron.state[step] * g.weight
                            for g in self.gates[GateDirection.OUT.value]
                            if g.to_neuron.state])
                       if step > -1
                       else 0)

        # Per evitare errori dati dal min() e max()
        if not states:
            states.append(0)

        states = self.aggregation(states)
        self.state.append(self.squash(states + prev_states + self.bias))

    def add_gate(self, neuron, weight):
        """Crea il gate da neurone a un altro.

        Args:
          to_neuron: L'istanza del neurone al quale assegnare il gate.
          weight: Il peso del gate, quasi certamente un float).

        Returns:
          L'istanza del gate.
        """
        gate = Synapse(self, neuron, weight)
        neuron.gates[GateDirection.IN.value].append(gate)
        self.gates[GateDirection.OUT.value].append(gate)

        return gate

    def sub_gate(self, neuron):
        """Rimuove il gate verso un neurone.

        Args:
          neuron: L'istanza del neurone al quale disconnetterlo.
        """
        for gate in self.gates[GateDirection.OUT.value]:
            if gate.to_neuron == neuron:
                gate.remove()
                break
