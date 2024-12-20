#!/bin/bash

python3.10 $1 > output.mlir
cat output.mlir
echo ""
echo "---------------------------"
echo ""
tt-opt output.mlir -t tt-metalium
