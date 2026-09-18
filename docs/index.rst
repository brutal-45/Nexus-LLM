Nexus-LLM documentation
=======================

**Nexus-LLM** is a terminal-based LLM chat application with its own local
inference backend: interactive chat, an OpenAI-compatible API server,
fine-tuning (LoRA/QLoRA), evaluation and benchmarking — all driven from a
single ``nexus-llm`` command.

.. code-block:: bash

   pip install -e ".[dev]"
   nexus-llm chat --model gpt2-medium

Getting started
---------------

.. toctree::
   :maxdepth: 2

   INSTALL
   TRAINING
   MODELS
   FAQ
   TROUBLESHOOTING
   SECURITY

Guides
------

.. toctree::
   :glob:
   :maxdepth: 2

   guides/*

Tutorials
---------

.. toctree::
   :glob:
   :maxdepth: 2

   tutorials/*

Architecture
------------

.. toctree::
   :glob:
   :maxdepth: 2

   architecture/*

API reference
-------------

.. toctree::
   :glob:
   :maxdepth: 2

   api/*
   API
   ARCHITECTURE

Module documentation is generated from the package docstrings:

Module reference
~~~~~~~~~~~~~~~~

Generated recursively from the package itself, so every public module, class
and function documented in the source appears here.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   nexus_llm

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Project
-------

.. toctree::
   :maxdepth: 1

   CHANGELOG
   CONTRIBUTING
