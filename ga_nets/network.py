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
"""Architettura portante della rete neurale"""
from ga_nets.connection import AlreadyConn, is_connected, SynapseDirection
from ga_nets.index import Indexer
import ga_nets.layer as Layer
from ga_nets.neuron import ErrNeuronType, NeuronType


class NeuronNotInConns(Exception):
    """Utilizzata per quando uno dei neuroni nella connessione non è presente
    nelle lista dei neuroni della rete neurale"""


class Network():
    """La classe che si occupa di gestire la struttura generale della rete
    neurale"""
    def __init__(self, traits, neuron_class):
        """Inizializza i dati utilizzati nel network.

        Args:
          traits: Dizionario con le impostazioni di default. Per esempio:
                    {"aggregation_fn": sum}
                  Dove 'sum' è la funzione di default in python oppure:
                    {"aggregation_fn": my_rnd_agg}
                  Dove 'my_rnd_agg' è una funzione che restituisce casualmente
                  una funzione di aggregazione che può essere:
                    'sum', 'mean', 'abs', etc...
          neuron_class: Classe del tipo di neurone: feedforward o recurrent.
        """
        self.neuron_class = neuron_class
        self.traits = traits

        self.__key = Indexer.get_id("network")

        self.__neurons = {}  # Istanze dei neuroni
        self.__synapses = []  # Istanze delle connessioni
        self.__layers = []

    def __str__(self):
        """Stampa le informazioni riguardante il network"""
        output = "Network #{}\n".format(self.__key)

        output += "   Neurons:\n"
        for neuron in self.neurons.values():
            output += "      {}\n".format(neuron)

        output += "   Layers:\n"
        output += "      {}\n".format(
            " -> ".join(map(str, [l for l in self.get_layers()])))

        if self.synapses:
            output += "   Synapses:\n"
            for synapse in self.synapses:
                output += "      {}\n".format(synapse)

        return output

    @property
    def neurons(self):
        """Getter per la prorietà dei neuroni.

        Returns:
          L'array con i neuroni.
        """
        return self.__neurons

    @neurons.setter
    def neurons(self, neurons):
        self.__neurons = neurons

    @property
    def synapses(self):
        """Getter per la prorietà delle connessioni.

        Returns:
          L'array con le connessioni.
        """
        return self.__synapses

    @synapses.setter
    def synapses(self, synapses):
        self.__synapses = synapses

    @property
    def layers(self):
        """Getter per la prorietà dei layers.

        Returns:
          La lista tridimensionale con i neuroni in ogni layer.
        """
        if not self.__layers:
            self.__layers = self.get_layers()

        return self.__layers

    @layers.setter
    def layers(self, layers):
        self.__layers = layers

    @property
    def num_inputs(self):
        """Getter per la prorietà del numero di inputs.

        Returns:
          Un intero con il numero di inputs.
        """
        return len(self.get_neuron_list(NeuronType.INPUT))

    @property
    def num_outputs(self):
        """Getter per la prorietà del numero di outputs.

        Returns:
          Un intero con il numero di inputs.
        """
        return len(self.get_neuron_list(NeuronType.OUTPUT))

    @property
    def num_hiddens(self):
        """Getter per la prorietà del numero di inputs.

        Returns:
          Un intero con il numero di inputs.
        """
        return len(self.get_neuron_list(NeuronType.HIDDEN))

    def activate(self, features):
        """Attiva la rete neurale.

        Args:
          features: Una matrice tridimensionale con gli input
                    (es. '[[0, 1]]' o '[[0, 0], [1, 1]]').

        Returns:
          Una matrice tridimensionali con i valori di output (es. '[[1]]' o
          '[[1], [1]]').

        Raises:
          ValueError: Se la dimensione dell'array degli input è errata.
        """
        if len(features[0]) != self.num_inputs:
            raise ValueError("Features' number is wrong.")

        # Resetta gli stati
        self.clear()

        # Costruisce i layer di neuroni
        neurons = [self.neurons[n] for l in self.layers for n in l]

        # Attiva i neuroni e restituisce gli output in ordine
        outputs = []
        for feature in features:
            for i, neuron in enumerate(neurons):
                if neuron.type is NeuronType.INPUT:
                    neuron.state.append(feature[i])
                else:
                    neuron.activate()

            outputs.append([self.neurons[o].state[-1]
                            for o in self.layers[-1]])

        return outputs

    def clear(self):
        """Resetta gli stati dei neuroni"""
        for neuron in self.neurons.values():
            neuron.state = []

    def add_neuron(self, **kwargs):
        """Aggiunge un nuovo neurone.

        Args:
          neuron_type: Il tipo di neurone: input, output o hidden, assegnato
                       tramite la classe NeuronType. Di default è 'hidden'.
          bias: Bias del neurone. Se non presente viene generato dalla
                funzione di default dei tratti.
          squash: Funzione d'attivazione. Se non presente viene assegnato
                  da quella di default nei tratti.
          aggregation: Funzione di aggregazione. Se non presente viene
                       assegnato da quella di default nei tratti.
          key: La chiave del neurone, se assente cerca l'indice maggiore e
               aggiunge un'unità.

        Returns:
          L'istanza del neurone.
        """
        if "key" not in kwargs:
            keys = self.neurons.keys()
            kwargs["key"] = max(map(int, keys)) + 1 if keys else 0
        if "bias" not in kwargs:
            kwargs["bias"] = self.traits["bias_fn"](self)
        if "squash" not in kwargs:
            kwargs["squash"] = self.traits["squash_fn"]
        if "aggregation" not in kwargs:
            kwargs["aggregation"] = self.traits["aggregation_fn"]
        if "neuron_type" not in kwargs:
            kwargs["neuron_type"] = NeuronType.HIDDEN

        neuron = self.neuron_class(key=kwargs["key"],
                                   neuron_type=kwargs["neuron_type"],
                                   bias=kwargs["bias"],
                                   squash=kwargs["squash"],
                                   aggregation=kwargs["aggregation"])
        self.neurons[kwargs["key"]] = neuron

        return neuron

    def sub_neuron(self, neuron):
        """Rimuove il neurone dal network.

        Args:
          neuron: Istanza del neurone da rimuovere.
        """
        del self.neurons[neuron.key]

        # Rimuove le connessioni
        for direction in SynapseDirection:
            for synapse in neuron.synapses[direction.value]:
                self.synapses.remove(synapse)

    def get_neuron_list(self, neuron_type):
        """Ritorna la lista di neuroni del network.

        Args:
          Il tipo di neurone.

        Returns:
          Una lista con tutte le istanze del neurone di quel tipo appartenenti
          al network.

        Raises:
          Se il tipo di neurone non fa parte della classe `NeuronType`.
        """
        if neuron_type not in NeuronType:
            raise ErrNeuronType("Invalid neuron type: {}".format(neuron_type))

        return [neuron
                for neuron in self.neurons.values()
                if neuron.type is neuron_type]

    def add_synapse(self, from_neuron, to_neuron, weight=None):
        """Connette due neuroni tramite sinapsi.

        Args:
          from_neuron: Istanza del neurone di partenza.
          to_neuron: Istanza del neurone d'arrivo.
          weight: Peso della connessione, se non definita viene assegnata
                  tramite la funzione assegnata nei 'traits'.
        Returns:
          L'istanza della connessione fra i due neuroni.

        Raises:
          AlreadyConn: Se la connessione è già presente.
        """
        if is_connected(self.synapses, from_neuron, to_neuron):
            AlreadyConn("Synapse already present.")

        if weight is None:
            weight = self.traits["weight_fn"](self)

        synapse = from_neuron.add_synapse(to_neuron, weight)
        self.synapses.append(synapse)

        return synapse

    def sub_synapse(self, synapse):
        """Rimuove una sinapsi fra i due neuroni.

        Args:
          synapse: Istanza della connessione.
        """
        self.synapses.remove(synapse)
        synapse.remove()

    def get_synapses(self):
        """Connessioni della rete neurale.

        Args:
          network: Istanza della rete neurale.

        Returns:
          Una lista con le tuple contenente come primo indice il neurone
          d'uscita e il secondo indice quello di entrata, es: '[(0, 3),
                                                                (1, 3),
                                                                (3, 2)]'.
        """
        return [(s.from_neuron.key,
                 s.to_neuron.key,
                 s.weight) for s in self.synapses]

    def get_layers(self):
        """Costruisce e ritorna i layers  con i corretti ordini d'attivazione

        Returns:
          Ipotizzando di avere una rete neurale composta da 3 neuroni tutti
          connessi:
            1) #0 Input
            2) #1 Output
            3) #2 Hidden

          Restituirà una lista come '[[0], [2], [1]]'.
        """
        inputs = [n.key for n in self.get_neuron_list(NeuronType.INPUT)]
        hiddens = [n.key for n in self.get_neuron_list(NeuronType.HIDDEN)]
        outputs = [n.key for n in self.get_neuron_list(NeuronType.OUTPUT)]

        conns = self.get_synapses()
        links = Layer.get_links(inputs, hiddens, outputs, conns)

        return Layer.get_layers(inputs, hiddens, outputs, conns, links)
