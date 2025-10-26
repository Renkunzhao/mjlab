Welcome to mjlab!
=================

.. figure:: source/_static/mjlab-banner.jpg
   :width: 100%            
   :alt: mjlab   

What is mjlab?
==============

**mjlab = Isaac Lab's API + MuJoCo's simplicity + GPU acceleration**

We took Isaac Lab's proven manager-based architecture and RL abstractions, 
then built them directly on MuJoCo Warp. No translation layers, no Omniverse 
overhead. Just fast, transparent physics.

License
=======

mjlab is licensed under the Apache License, Version 2.0. Please refer 
to `License <https://github.com/mujocolab/mjlab/blob/main/LICENSE/>`_ for more details

Acknowledgment
==============

mjlab wouldn't exist without the excellent work of the Isaac 
Lab team, whose API design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and 
Taylor Howell — for answering our questions, giving helpful 
feedback, and implementing features based on our requests 
countless times.

If you used mjlab in your research, we would appreciate it if you could cite it:

.. code:: bibtex

    @software{Zakka_MJLab_Isaac_Lab_2025,
        author = {Zakka, Kevin and Yi, Brent and Liao, Qiayuan and Le Lay, Louis},
        license = {Apache-2.0},
        month = sep,
        title = {{MJLab: Isaac Lab API, powered by MuJoCo-Warp, for RL and robotics research.}},
        url = {https://github.com/mujocolab/mjlab},
        version = {0.1.0},
        year = {2025}
    }


Table of Contents
=================

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   source/starting/installation_guide
   source/starting/migration_guide
   source/starting/create_new_task

.. toctree::
   :maxdepth: 1
   :caption: Overview

   source/overview/motivation
   source/overview/faq

.. toctree::
   :maxdepth: 1
   :caption: Core Concepts

   source/core/domain_randomization
   source/core/nan_guard
   source/core/obs_history_delay
   source/core/spec_config

.. toctree::
   :maxdepth: 1
   :caption: Source API

   source/api/index