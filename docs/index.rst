.. KCIF documentation master file, created by
   sphinx-quickstart on Sat Oct 19 14:55:39 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

KCIF documentation
===========================

KCIF is a benchmark for evaluating the instruction-following capabilities of Large Language Models (LLM). We adapt existing knowledge benchmarks and augment them with instructions that are a) conditional on correctly answering the knowledge task or b) use the space of candidate options in multiple-choice knowledge-answering tasks. KCIF allows us to study model characteristics, such as their change in performance on the knowledge tasks in the presence of answer-modifying instructions and distractor instructions.

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   introduction

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 1
   :caption: Dataset Conversion

   data_creation_rst

.. toctree::
   :maxdepth: 1
   :caption: Instruction Creation Guidelines

   instruction_creation_rst

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   modules

.. toctree::
   :maxdepth: 1
   :caption: Citation:

   citation
