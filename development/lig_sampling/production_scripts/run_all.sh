#!/bin/bash

# Use mpiexec to run locally.
mpiexec -n 12 python -m mpi4py.futures mpirun.py