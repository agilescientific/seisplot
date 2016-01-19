seisplot
========

Get the repo and enter that directory.

First make and enter an environment::

    conda create -n seisplot python=3.4
    source activate seisplot

Then install some more dependencies::

    conda install -c obspy obspy
    conda install nose
    conda install pyyaml

Then edit `config.yaml` to meet your requirements.

Run the script from the command line::

    python seisplot.py --config config.yaml <infile.segy> <outfile.png>
