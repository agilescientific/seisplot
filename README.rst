seisplot
========

If you don't already have a reliable Python installation, and know how to wield it, I recommend downloading and installing `Anaconda <https://www.continuum.io/downloads>`_.

Get this repo with ``git`` or by downloading the ZIP file, and enter that directory.

Make and enter a virtual environment::

    conda create -n seisplot python=3.4
    source activate seisplot

Install some more dependencies::

    conda install -c obspy obspy
    conda install nose
    conda install pyyaml

Edit ``config.yaml`` to meet your requirements.

Run the script from the command line, for example::

    python seisplot.py </path/to/infile.segy> --config config.yaml --out /path/to/result.pdf
