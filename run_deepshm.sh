#!/bin/sh

julia --project=@. -e 'include("src/DeepSHM.jl")' -- $@
