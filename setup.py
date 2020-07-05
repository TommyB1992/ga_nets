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
"""File for packaging with pip"""
import setuptools
from setuptools import setup

setup(name="Genetic Algorithm Networks",
      version="1.0.0",
      description="Neural Networks with a dynamic structure for"
                  "Genetic Algorithms.",
      author="Tomas Bartoli",
      author_email="tomas.bartoli@transcorp.org",
      license="MIT",
      packages=setuptools.find_packages(),
      zip_safe=False)
