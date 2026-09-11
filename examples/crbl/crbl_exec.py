#!/bin/sh
"exec" "$DGD/lib/.crbl/bin/python3" "-B" "$0" "$@"

from os.path import dirname, join
from crbl import main
from sys import path

if __name__ == "__main__":
    BASE = dirname(dirname(__file__))
    path.insert(0, join(BASE, "lib"))
    
    main()