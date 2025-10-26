.. _installation_guide:

Installation Guide
==================

System Requirements
-------------------

The basic requirements are:

- **Python:** 3.10 or higher
- **Operating System:** 
    - Linux (recommended)
    - macOS (limited support - see note below)
    - Windows (untested)
- **GPU:** NVIDIA GPU strongly recommended 
    - **CUDA Compatibility:** Not all CUDA versions are supported by MuJoCo Warp 
        - Check https://github.com/google-deepmind/mujoco_warp/issues/101 for CUDA version compatibility
        - **Recommended**: CUDA 12.4+ (for `conditional control flow <https://nvidia.github.io/warp/modules/runtime.html#conditional-execution/>`_ in CUDA graphs)


.. attention::
    
    mjlab is designed for large-scale training in GPU-accelerated simulations. 
    Since macOS does not support GPU acceleration, it is **not recommended** 
    for training. Even policy evaluation runs significantly slower on macOS. 
    We are working on improving this with a C-based MuJoCo backend for 
    evaluation — stay tuned for updates.


.. attention::

    mjlab is currently in **beta**. Expect frequent breaking changes in the coming weeks.
    There is **no stable release yet**.

    - The first beta snapshot is available on PyPI.
    - **Recommended**: install from source (or Git) to stay up-to-date with fixes and improvements.

Prerequisites
-------------

If you haven't already installed `uv <https://docs.astral.sh/uv/>`_, run:

.. code-block:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh


Installation Methods
--------------------

Method 1: From Source (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use this method if you want the latest beta updates.


.. tab-set::
   :sync-group: os

   .. tab-item:: Local editable install
      :sync: editable

      Clone the repository:

      .. code:: bash

         git clone https://github.com/mujocolab/mjlab.git && cd mjlab


      Add as an editable dependency to your project:

      .. code:: bash

         uv add --editable /path/to/cloned/mjlab


   .. tab-item:: Direct git install
      :sync: git

      Install directly from GitHub without cloning:

      .. code:: bash

         uv add "mjlab @ git+https://github.com/mujocolab/mjlab" "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@486642c3fa262a989b482e0e506716d5793d61a9"


      .. note::
            
        ``mujoco-warp`` must be installed from Git since it's not available on PyPI.


Method 2: From PyPI (Beta Snapshot)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can install the latest beta snapshot from PyPI, but note:
- It is **not stable**
- You still need to install ``mujoco-warp`` from Git
  
.. code-block:: bash

    uv add mjlab "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@486642c3fa262a989b482e0e506716d5793d61a9"


Method 3: Using pip (venv, conda, virtualenv, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While mjlab is designed to work with `uv <https://docs.astral.sh/uv/>`_, you can
also use it with any pip-based virtual environment (venv, conda, virtualenv, etc.).

Create and activate your virtual environment:

  .. tab-set::
     :sync-group: os

     .. tab-item:: venv
        :sync: venv

        .. code:: bash

           python -m venv mjlab-env
           source mjlab-env/bin/activate


     .. tab-item:: conda
        :sync: conda

        .. code:: bash

           conda create -n mjlab python=3.13
           conda activate mjlab


Install mjlab and dependencies via pip:

.. tab-set::
   :sync-group: os

   .. tab-item:: From source (recommended)
      :sync: source

      .. code:: bash

         pip install git+https://github.com/google-deepmind/mujoco_warp@486642c3fa262a989b482e0e506716d5793d61a9
         git clone https://github.com/mujocolab/mjlab.git && cd mjlab
         pip install -e .


      .. note::
            
        You must install ``mujoco-warp`` from Git before running ``pip install -e .`` 
        since it's not available on PyPI and pip cannot resolve the Git dependency 
        specified in ``pyproject.toml`` (which uses uv-specific syntax).


   .. tab-item:: From PyPI
      :sync: pypi

      .. code:: bash

         conda create -n mjlab python=3.13
         conda activate mjlab

Verification
------------

After installation, verify that mjlab is working by running the demo:

.. tab-set::
   :sync-group: os

   .. tab-item:: uv from source
      :sync: source

      If working inside the mjlab directory with uv.

      .. code:: bash

         uv run demo


   .. tab-item:: uv from dependency
      :sync: dependency

      If mjlab is installed as a dependency in your project with uv.

      .. code:: bash

         uv run python -m mjlab.scripts.demo

   .. tab-item:: CLI command
      :sync: pypi

      If installed via pip (conda, venv, etc.), use the CLI command directly.

      .. code:: bash

         demo

   .. tab-item:: Module syntax
      :sync: pypi

      Works anywhere mjlab is installed (module syntax).

      .. code:: bash

         python -m mjlab.scripts.demo



Troobleshooting
---------------

If you run into problems:

1. **Check the FAQ**: [faq.md](faq.md) may have answers to common issues.
2. **CUDA Issues**: Verify your CUDA version is supported by MuJoCo Warp
   `see compatibility list <https://github.com/google-deepmind/mujoco_warp/issues/101>`_.
3. **macOS Slowness**: Training is not supported; evaluation may still be slow
   (see macOS note above).
4. **Still stuck?** Open an issue on
   `GitHub Issues <https://github.com/mujocolab/mjlab/issues>`_.

