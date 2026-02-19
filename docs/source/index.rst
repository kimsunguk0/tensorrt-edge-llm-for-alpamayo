.. TensorRT Edge-LLM documentation master file, created by
   sphinx-quickstart on Wed Oct  8 16:38:08 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TensorRT Edge-LLM Documentation
================================

Welcome to the TensorRT Edge-LLM documentation. This library provides optimized inference capabilities 
for large language models and vision-language models on edge devices.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   developer_guide/getting-started/overview.md
   developer_guide/getting-started/supported-models.md
   developer_guide/getting-started/installation.md
   developer_guide/getting-started/quick-start-guide.md
   developer_guide/getting-started/examples.md
   developer_guide/getting-started/input-format.md
   developer_guide/getting-started/chat-template-format.md

.. toctree::
   :maxdepth: 2
   :caption: Software Design

   developer_guide/software-design/python-export-pipeline.md
   developer_guide/software-design/engine-builder.md
   developer_guide/software-design/cpp-runtime-overview.md
   developer_guide/software-design/llm-inference-runtime.md
   developer_guide/software-design/llm-inference-specdecode-runtime.md

.. toctree::
   :maxdepth: 2
   :caption: Advanced Features

   developer_guide/features/lora.md
   developer_guide/features/reduce-vocab.md
   developer_guide/features/FP8KV.md
   developer_guide/features/system-prompt-cache.md

.. toctree::
   :maxdepth: 2
   :caption: Customization

   developer_guide/customization/customization-guide.md
   developer_guide/customization/tensorrt-plugins.md

.. toctree::
   :maxdepth: 2
   :caption: APIs

   python_api
   cpp_api

----

**Need help?** Visit our `GitHub repository <https://github.com/NVIDIA/TensorRT-Edge-LLM>`_ for issues and discussions.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
