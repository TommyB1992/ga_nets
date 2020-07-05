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
"""Connessioni fra neuroni (gates o sinapsi)"""
from abc import ABCMeta, abstractmethod

from enum import Enum


class AlreadyConn(Exception):
    """Errore di quando una connessione è già presente"""


class ConnDirection(Enum):
    """Direzione della connessione"""
    IN = 0
    OUT = 1


# Crea degli alias semplicemente per avere un nome differente
SynapseDirection = ConnDirection
GateDirection = ConnDirection


class Connection(metaclass=ABCMeta):
    """Base astratta utilizzata per sinapsi e gates"""
    def __init__(self, from_neuron, to_neuron, weight):
        """Inizializza la classe.

        Args:
          from_node: Istanza del neurone di provenienza.
          to_neuron: Istanza del neurone d'arrivo.
          weight: Peso della connessione, probabilmente un float.
        """
        self.__from_neuron = from_neuron
        self.__to_neuron = to_neuron
        self.__weight = weight

    def __str__(self):
        """Stampa le informazioni riguardo la connessione"""
        return "From {} to {} (w: {})".format(self.from_neuron.key,
                                              self.to_neuron.key,
                                              self.weight)

    @property
    def from_neuron(self):
        """Non potendo variare ha solo l'attibuto getter che restituisce
        l'istanza del neurone di provenienza"""
        return self.__from_neuron

    @property
    def to_neuron(self):
        """Non potendo variare ha solo l'attibuto getter che restituisce
        l'istanza del neurone d'arrivo"""
        return self.__to_neuron

    @property
    def weight(self):
        """Getter del peso della connessione.

        Returns:
          Float o, improbabilmente, un intero."""
        return self.__weight

    @weight.setter
    def weight(self, weight):
        self.__weight = weight

    @abstractmethod
    def remove(self):
        """Rimuove la connessione, da implementare."""
        raise NotImplementedError("Needs to be implemented.")


class Synapse(Connection):
    """Sinapsi fra i neuroni del network."""
    def remove(self):
        """Elimina le connessioni dalle liste dei numeroni"""
        self.from_neuron.synapses[SynapseDirection.OUT.value].remove(self)
        self.to_neuron.synapses[SynapseDirection.IN.value].remove(self)


class Gate(Synapse):
    """Gates fra i neuroni del network"""
    def remove(self):
        """Elimina i gates dalle liste dei neuroni"""
        self.from_neuron.gates[GateDirection.OUT.value].remove(self)
        self.to_neuron.gates[GateDirection.IN.value].remove(self)


def is_connected(conns, from_neuron, to_neuron):
    """Verifica se c'è già una connessione (Sinpasi o Gates).

    Args:
      conns: Lista con sinapsi o gates.
      from_neuron: Istanza del neurone di partenza.
      to_neuron: Istanza del neurone d'arrivo.

    Returns:
      True se è già presente, False altrimenti."""
    for conn in conns:
        if conn.from_neuron == from_neuron and conn.to_neuron == to_neuron:
            return True

    return False
