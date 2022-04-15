#!/bin/bash

mkdir -p $PREFIX/bin

# Conda package requires chopin.py and functions.py scripts only
# Everything else will be removed
find . -type f \( -iname "chopin.py" -o -iname "functions.py" \) -exec cp {} $PREFIX/bin \;

# Make Python scripts executable
chmod +x $PREFIX/bin/hdclass.py $PREFIX/bin/functions.py
