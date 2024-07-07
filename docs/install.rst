Installation
============

.. note::

    Wheels are provided for x86-64 Linux linking against OpenBLAS. Other 
    machines will have to build the wheel from the source distribution. 
    Building ``rlinalg`` involves compiling Fortran code, and a Fortran
    compiler is needed to build from source.


PyPi
^^^^

``rlinalg`` is hosted on GitHub, but the easiest way to install it is to download
the latest release from its `PyPi repository <https://pypi.python.org/pypi/rlinalg>`_.
It will install all build dependencies then install ``rlinalg`` either from 
a wheel if one is available, or from source after compiling the Fortran code :

.. code:: console

   $ pip install --user rlinalg


.. Arch User Repository
.. ^^^^^^^^^^^^^^^^^^^^

.. A package recipe for Arch Linux can be found in the Arch User Repository
.. under the name `python-diced <https://aur.archlinux.org/packages/python-rlinalg>`_.
.. It will always match the latest release from PyPI.

.. Steps to install on ArchLinux depend on your `AUR helper <https://wiki.archlinux.org/title/AUR_helpers>`_
.. (``yaourt``, ``aura``, ``yay``, etc.). For ``aura``, you'll need to run:

.. .. code:: console

..     $ aura -A python-rlinalg


Piwheels
^^^^^^^^

``rlinalg`` is compatible with Raspberry Pi computers, and pre-built
wheels are compiled for `armv7l` platforms on `piwheels <https://www.piwheels.org>`_.
Run the following command to install these instead of compiling from source:

.. code:: console

   $ sudo apt install libgfortran5 libopenblas0-pthread
   $ pip3 install rlinalg --extra-index-url https://www.piwheels.org/simple

Check the `piwheels documentation <https://www.piwheels.org/faq.html>`_ 
and the `rlinalg page on piwheels <https://www.piwheels.org/project/rlinalg/>`_ 
for more information.


GitHub + ``pip``
^^^^^^^^^^^^^^^^

If, for any reason, you prefer to download the library from GitHub, you can clone
the repository and install the repository by running (with the admin rights):

.. code:: console

   $ git clone --recursive https://github.com/althonos/rlinalg
   $ pip install --user ./rlinalg

.. caution::

    Keep in mind this will install always try to install the latest commit,
    which may not even build, so consider using a versioned release instead.

