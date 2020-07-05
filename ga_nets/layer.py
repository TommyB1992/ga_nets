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
"""Funzioni per la costruzione dei layer di neuroni"""

# Qualasiasi ciclo che supera questo numero di iterazioni, vuol dire che ha
# una logica di programmazione errata.
MAX_ITERS = 10000000


class LoopError(RuntimeWarning):
    """Rilanciata quando un ciclo diventa infinito per errori di
    programmazione"""
    def __init__(self, neurons, conns, links):
        super().__init__("Over {} loops. Variables:\n"
                         "  inputs = {}\n"
                         "  hiddens = {}\n"
                         "  outputs = {}\n"
                         "  conns = {}\n"
                         "  links = {}".format(MAX_ITERS,
                                               neurons[0],
                                               neurons[1],
                                               neurons[2],
                                               conns,
                                               links))


def remove_void_links(links, inputs, hiddens, outputs):
    """Rimozione dei neuroni che non sono utili per il computo di nessun altro
    neurone.

    Args:
      links: Dizionario dal quale rimuovere i neuroni inutili.
      inputs: Lista con le chiavi dei neuroni di inputs.
      hiddens: Lista con le chiavi dei neuroni di hiddens.
      outputs: Lista con le chiavi dei neuroni di outputs.

    Returns:
      True se il ciclo ha superato 10M di iterazioni (e quindi c'è un errore
      di programmazione), False altrimenti.
    """
    for output in outputs:
        links[output] = []

    neurons = inputs + hiddens + outputs
    for _ in range(MAX_ITERS):
        removed = False
        for neuron in neurons:
            try:
                if links[neuron]:
                    continue
                removed = True
                del links[neuron]
                for neuron2 in neurons:
                    if neuron2 in links and neuron in links[neuron2]:
                        links[neuron2].remove(neuron)
            except KeyError:
                continue
        if not removed:
            return False

    return True


def disjointed_links(inputs, hiddens, outputs, links, conns):
    """Aggiunge i links che non è riuscita a calcolare nella funzione
    principale 'get_links()'.

    Args:
      inputs: Lista con le chiavi dei neuroni di inputs.
      hiddens: Lista con le chiavi dei neuroni di hiddens.
      outputs: Lista con le chiavi dei neuroni di outputs.
      links: Links con le connessioni, essendo un dizionario viene aggionato.
      conns: Lista con le tuple contenente come primo indice il neurone
             d'uscita, il secondo indice quello di entrata e il terzo il
             peso, che però non ha nessuna utilità. Es: `[(0, 3, peso1),
                                                          (1, 3, peso2),
                                                          (3, 2, peso3)]`.
    """
    neurons = inputs + hiddens + outputs
    for neuron in neurons:
        if neuron in outputs or neuron in inputs:
            continue

        links[neuron] = []
        for conn in conns:
            if neuron == conn[1]:
                links[neuron].append(conn[0])


def get_links(inputs, hiddens, outputs, conns):
    """Links fra neuroni.

    Args:
      inputs: Lista con le chiavi dei neuroni di inputs.
      hiddens: Lista con le chiavi dei neuroni di hiddens.
      outputs: Lista con le chiavi dei neuroni di outputs.
      conns: conns: Lista con le tuple contenente come primo indice il neurone
             d'uscita, il secondo indice quello di entrata e il terzo il
             peso, che però non ha nessuna utilità. Es: '[(0, 3, peso1),
                                                          (1, 3, peso2),
                                                          (3, 2, peso3)]'.

    Returns:
      Un dizionario nel quale la chiave è il neurone d'entrata mentre il
      valore è una lista con le chiavi dei neuroni d'uscita a quel neurone.

    Raises:
      LoopError: Se i parametri passati sono errati e il ciclo supera le 10M
                 di iterazioni solleva un errore (nessuna rete neurale può
                 essere tanto grande da necessitare tante iterazioni).
    """
    neurons = inputs + hiddens + outputs
    curr_neurons = outputs

    links = {}

    for _ in range(MAX_ITERS):
        next_neurons = []
        for neuron in curr_neurons:
            for conn in conns:
                if neuron != conn[1]:
                    continue

                if neuron not in links:
                    links[neuron] = []
                    neurons.remove(neuron)
                if conn[0] not in links[neuron]:
                    links[neuron].append(conn[0])
                if conn[0] not in next_neurons and conn[0] not in inputs:
                    next_neurons.append(conn[0])

        if not next_neurons:
            disjointed_links(inputs, hiddens, outputs, links, conns)

            # 'remove_void_links()' ritorna True solo se il ciclo ha superato
            # le 10M di iterazioni.
            if links and remove_void_links(links, inputs, hiddens, outputs):
                break

            return links

        curr_neurons = next_neurons

    raise LoopError((inputs, hiddens, outputs), conns, links)


def is_in_prev_layers(conn, links, layers):
    """Controlla se la connessione è presente nei layers precedenti.

    Args:
      conn: Il neurone da controllare.
      links: Dizionario con i links fra neuroni (chi è collegato con chi)
             generato da 'get_links()'.
      layers: Attuali layers di neuroni.

    Returns:
      True se è presente, False altrimenti."""
    if conn not in links:
        return False

    for required in links[conn]:
        try:
            if not links[required]:
                continue
        except KeyError:
            pass

        found = False
        for layer in layers:
            if required in layer:
                found = True
                break

        if not found:
            return False

    return True


def get_layers(inputs, hiddens, outputs, conns, links):
    """Restituisce i layers corretti con l'ordine d'attivazione dei neuroni.

    Args:
      inputs: Lista con le chiavi dei neuroni di inputs.
      hiddens: Lista con le chiavi dei neuroni di hiddens.
      outputs: Lista con le chiavi dei neuroni di outputs.
      conns: Lista con le tuple contenente come primo indice il neurone
             d'uscita, il secondo indice quello di entrata e il terzo il
             peso, che però non ha nessuna utilità. Es: '[(0, 3, peso1),
                                                          (1, 3, peso2),
                                                          (3, 2, peso3)]'.
      links: Dizionario con i links fra neuroni (chi è collegato con chi)
             generato da 'get_links()'.

    Returns:
      Una lista tridimensionale, dove il primo indice contiene i neuroni di
      input, l'ultimo quelli di output e quelli in mezzo sono gli hidden layer
      ordinati.

    Raises:
      LoopError: Se i parametri passati sono errati e il ciclo supera le 10M
                 di iterazioni solleva un errore (nessuna rete neurale può
                 essere tanto grande da necessitare tante iterazioni).
    """
    links = get_links(inputs, hiddens, outputs, conns)

    layers = [inputs[:]]
    for _ in range(MAX_ITERS):
        layer = []

        for gene in layers[-1]:
            for conn in conns:
                if conn[1] in outputs:
                    continue

                if not is_in_prev_layers(conn[1], links, layers):
                    continue

                if (conn[0] == gene
                        and gene in links[conn[1]]
                        and conn[1] not in layer):
                    layer.append(conn[1])

        if not layer:
            layers.append(outputs[:])
            return layers

        layers.append(layer)

    raise LoopError((inputs, hiddens, outputs), conns, links)
