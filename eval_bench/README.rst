Evaulation Benchmark for SSL
============================

|Build Status| |Python 3.10+|

Benchopt is a package to simplify and make more transparent and
reproducible comparisons of ML methods on multiple datasets.
This benchmark is dedicated to evaluting self-supervised models learned
with SynthSeg on tasks from openBHB.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/Duplums/2026_synthseg_ssl_mind_datathon
   $ benchopt run 2026_synthseg_ssl_mind_datathon

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run 2026_synthseg_ssl_mind_datathon -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/Duplums/2026_synthseg_ssl_mind_datathon/workflows/Tests/badge.svg
   :target: https://github.com/Duplums/2026_synthseg_ssl_mind_datathon/actions
.. |Python 3.10+| image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/release/python-310/