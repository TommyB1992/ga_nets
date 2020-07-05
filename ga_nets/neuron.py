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
"""Architettura base dei neuroni (che possono essere per i feedforward o
recurrent"""
from abc import ABCMeta, abstractmethod
from enum import Enum

from ga_nets.connection import Synapse, SynapseDirection


def from_fn_to_str(fn_name):
    """Rende leggibile il nome della funzione.

    Args:
      L'handle della funzione."""
    return (str(fn_name).split("function ")[1]
            .split(" ")[0]
            .replace(">", "")
            .split(".")[0])


class ErrNeuronType(TypeError):
    """Utilizzato quando il tipo del neurone non è valido"""


class NeuronType(Enum):
    """Tipo di neurone"""
    INPUT = 0
    OUTPUT = 1
    HIDDEN = 2


class Neuron(metaclass=ABCMeta):
    """Neurone utilizzato nella rete neurale"""
    def __init__(self, **kwargs):
        self.__type = (kwargs["neuron_type"]
                       if "neuron_type" in kwargs
                       else NeuronType.HIDDEN)

        if self.__type not in NeuronType:
            raise ErrNeuronType("Invalid neuron type: {}".format(self.__type))

        self.__key = kwargs["key"]
        self.__bias = kwargs["bias"]
        self.__squash = kwargs["squash"]
        self.__aggregation = kwargs["aggregation"]

        # E' un array in quanto utilizzato anche da RNN.
        # Nel caso venga utilizzato da un feedforward
        # network avrà un solo elemento.
        self.__state = []

        self.__synapses = [[] for _ in SynapseDirection]

    def __str__(self):
        """zzz"""
        squash_name = from_fn_to_str(self.squash)
        aggregation_name = from_fn_to_str(self.aggregation)
        return "#{} (t: {}, b: {}, s: {}, a: {})".format(self.key,
                                                         self.type.name,
                                                         self.bias,
                                                         squash_name,
                                                         aggregation_name)

    @property
    def key(self):
        """Getter per l'indice del neurone.

        Returns:
          L'intero con l'indice."""
        return self.__key

    @key.setter
    def key(self, key):
        self.__key = key

    @property
    def type(self):
        """Ha solo il metodo getter in quanto non può variare il tipo.

        Returns:
          Un'istanza dell'oggetto Enum che definisce il tipo:
            - INPUT
            - HIDDEN
            - OUTPUT"""
        return self.__type

    @property
    def bias(self):
        """Getter per il bias del neurone.

        Returns:
          Un float con il bias del neurone."""
        return self.__bias

    @bias.setter
    def bias(self, bias):
        self.__bias = bias

    @property
    def squash(self):
        """Getter per la funzione d'attivazione.

        Returns:
          L'istanza della funzione utilizzata per l'attivazione."""
        return self.__squash

    @squash.setter
    def squash(self, squash):
        self.__squash = squash

    @property
    def aggregation(self):
        """Getter per la funzione d'aggreazione.

        Returns:
          L'istanza della funzione utilizzata per l'aggregazione."""
        return self.__aggregation

    @aggregation.setter
    def aggregation(self, aggregation):
        self.__aggregation = aggregation

    @property
    def state(self):
        """Getter che restituisce lo stato (o gli stati) del neurone.

        Returns:
          Un'array vuota o con gli interi/reali degli o dello stato
          d'attivazione."""
        return self.__state

    @state.setter
    def state(self, state):
        self.__state = state

    @property
    def synapses(self):
        """Getter che restituisce le connessioni fra i neuroni.

        Returns:
          Un'array vuota o con le istanze delle connessioni"""
        return self.__synapses

    @synapses.setter
    def synapses(self, synapses):
        self.__synapses = synapses

    @abstractmethod
    def activate(self):
        """Calcola lo stato del neurone"""
        raise NotImplementedError("Needs to be implemented.")

    def add_synapse(self, neuron, weight):
        """Connette questo neurone a un altro tramite sinapsi.

        Args:
          neuron: L'istanza del neurone al quale connetterlo.
          weight: Il peso della connessione, quasi certamente un float).

        Returns:
          L'istanza della connessione.
        """
        synapse = Synapse(self, neuron, weight)
        neuron.synapses[SynapseDirection.IN.value].append(synapse)
        self.synapses[SynapseDirection.OUT.value].append(synapse)

        return synapse

    def sub_synapse(self, neuron):
        """Rimuove la connessione tramite sinapsi da questo neurone verso un
        altro.

        Args:
          neuron: L'istanza del neurone al quale disconnetterlo.
        """
        for synapse in self.synapses[SynapseDirection.OUT.value]:
            if synapse.to_neuron == neuron:
                synapse.remove()
                break

    def is_projecting_to(self, neuron):
        """Verifica se il neurone ha una connessione in uscita verso un altro.

        Args:
          neuron: L'istanza della classe del neurone in entrata.

        Returns:
          True se la connessione è presente, False altrimenti.
        """
        for connection in self.synapses[SynapseDirection.OUT.value]:
            if neuron == connection.to_neuron:
                return True

        return False

    def copy(self):
        """Restituisce un oggetto identico a questa imitando la funzione per
        copiare, ma in maniera ottimizzata.

        Returns:
          Un'istanza dell'oggetto identico a questo stesso."""
        return self.__class__(key=self.key,
                              bias=self.bias,
                              squash=self.squash,
                              aggregation=self.aggregation,
                              neuron_type=self.type)


IO_TYPE = (NeuronType.INPUT, NeuronType.OUTPUT)
