#!/bin/bash

prep-gas ../hydrogen-ignition-delay/Jachimowski-1992-species.inp Jachimowski-1992-gas-model.lua
prep-chem Jachimowski-1992-gas-model.lua ../hydrogen-ignition-delay/Jachimowski-1992.lua Jachimowski-1992-reac-file.lua

python3 reaction-rates.py

