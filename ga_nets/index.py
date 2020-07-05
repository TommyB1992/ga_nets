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
"""Indexing di networks e neuroni"""
from itertools import count


class Indexer():
    """Gestisce le chiavi """
    indexes = {}

    @classmethod
    def get_id(cls, key):
        """Restituisce un indice.

        Args:
          key: Il nome della chiave del dizionario.

        Returns:
          Il contatore incrementato di una unità.
        """
        if key not in cls.indexes:
            cls.reset(key)

        return next(cls.indexes[key])

    @classmethod
    def reset(cls, key):
        """Resetta il contatore.

        Args:
          key: Il nome della chiave del dizionario dove resettare il
               contatore.
        """
        cls.indexes[key] = count(0)
