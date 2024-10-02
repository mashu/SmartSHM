# SmartSHM

[![Build Status](https://github.com/mashu/SmartSHM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mashu/SmartSHM.jl/actions/workflows/CI.yml?query=branch%3Amain)

Julia implementation using Flux of DeepSHM model from ["Deep learning model of somatic hypermutation reveals importance of sequence context beyond hotspot targeting" Catherine Tang et al 2022](https://doi.org/10.1016/j.isci.2021.103668).

## Prerequirement is Julia installation
Julia can be installed with [juliaup](https://github.com/JuliaLang/juliaup)
```
curl -fsSL https://install.julialang.org | sh
```
## Usage
Run

```
julia --project=@. -e 'using Pkg; Pkg.instantiate(); include("src/DeepSHM.jl")'
```
