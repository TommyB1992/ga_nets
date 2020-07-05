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
"""Testa le Recurrent Neural Network"""
from ga_nets.neuron import NeuronType
from ga_nets.nets.rnn import Recurrent
from ml_stats.activation import identity


def create_topology(network, neurons):
    """Funzione per creare la topologia per evitare di avere codice
    ridondante.

    Args:
      network: L'istanza della rete neurale.
      neurons: Una lista con i tipi di neuroni da creare.

    Returns:
      Una lista con le istanze dei neuroni creati.
    """
    return [network.add_neuron(key=i,
                               neuron_type=nt,
                               bias=0,
                               squash=identity,
                               aggregation=sum)
            for i, nt in enumerate(neurons)]


def connect_synapses(network, neurons, conns):
    """Connette le sinapsi della rete neurale.

    Args:
      network: L'istanza della rete neurale.
      neurons: Una lista con i tipi di neuroni da creare.
      conns: Una lista contenente le tuple con l'indice del neurone di
             partenza, il neurone d'arrivo e il peso della sinapsi.
    """
    for (neuron1, neuron2, weight) in conns:
        network.add_synapse(neurons[neuron1], neurons[neuron2], weight)


def connect_gates(network, neurons, gates):
    """Connette i gates della rete neurale.

    Args:
      network: L'istanza della rete neurale.
      neurons: Una lista con i tipi di neuroni da creare.
      gates: Una lista contenente le tuple con l'indice del neurone di
             partenza, il neurone d'arrivo e il peso della sinapsi.
    """
    for (neuron1, neuron2, weight) in gates:
        network.add_gate(neurons[neuron1], neurons[neuron2], weight)


def check_result(result, expected, margin):
    """Verifica i risultati della rete neurale.

    Args:
      result: Il risultato dato dalla rete neurale.
      expected: Il valore atteso.
      margin: Il margine d'errore permesso.
    """
    assert abs(result - expected) < margin, \
        "The result is {}, but needs to be {}.".format(result, expected)


def test_fully_connected_rnn():
    """Testa una comune rete neurale ricorrente con un solo hidden layer"""
    network = Recurrent({})
    neurons = create_topology(network, (NeuronType.INPUT,
                                        NeuronType.INPUT,
                                        NeuronType.OUTPUT,
                                        NeuronType.HIDDEN,
                                        NeuronType.HIDDEN))

    connect_synapses(
        network,
        neurons,
        [(0, 3, .1), (0, 4, .2), (1, 3, .3), (1, 4, .4),
         (3, 2, .5), (4, 2, .6)])

    connect_gates(network, neurons, [(3, 4, .1), (4, 3, .2),
                                     (3, 3, .3), (4, 4, .4)])

    result = network.activate([[1, 1], [1, 1]])

    # I risultati sono calcolati manualmente
    check_result(result[0][0], .56, .01)
    check_result(result[1][0], .842, .0001)


def test_no_conn_rnn():
    """Testa una rete neurale senza connessioni ma con dei gates"""
    network = Recurrent({})
    neurons = create_topology(network, (NeuronType.INPUT,
                                        NeuronType.INPUT,
                                        NeuronType.OUTPUT,
                                        NeuronType.HIDDEN,
                                        NeuronType.HIDDEN))

    connect_gates(network, neurons, [(3, 4, .1), (4, 3, .2),
                                     (3, 3, .3), (4, 4, .4)])

    result = network.activate([[1, 1], [1, 1]])

    # I risultati devono essere tutti zero in quanto l'output non è connesso
    check_result(result[0][0], 0, 0)
    check_result(result[1][0], 0, 0)


if __name__ == "__main__":
    test_fully_connected_rnn()
    test_no_conn_rnn()
