#!/bin/bash

PYTHONPATH=$(pwd) .venv/bin/python $1 > output.mlir
cat output.mlir
echo ""
echo "---------------------------"
echo ""
PYTHONPATH=$(pwd) tenstorrent/tools/tt-opt output.mlir -t tt-metalium
