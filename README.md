# HPAC-ML
HPAC-ML is a directive-based programming model that enables easy use of ML surrogate models in scientific applications. 
The programming model can invoke a model, replacing parts of an application with NN inference, or collect data during application execution to be used during offline training of ML models.


## Installation
We recommend using or building the provided [HPAC-ML container](https://github.com/ZwFink/hpacml_artifact).

Otherwise, you can install the spack environment from `spack.yaml`.

### Build HPAC Compiler Extensions
To build the HPAC-ML compiler and runtime system execute the following commands:

```bash
git clone git@github.com:LLNL/HPAC.git
cd HPAC
./setup.sh 'PREFIX' 'NUM THREADS' 
```

The 'PREFIX' argument defines where to install all the HPAC related binaries and executables. The 'NUM THREADS' parameter
define how many threads should the installation use. The installation script performs the following actions:

1. Configures, builds and installs Clang/LLVM including the approximation extensions.
2. Configures, builds and installs the approximation library. 
3. Creates a file, called 'hpac_env.sh', at the root of the project which should always be sourced before using HPAC.

The installation process can take a considerable amount of time. 
For quick exploration, we recommend 

## Contributing
To contribute to this repo please open a [pull
request](https://help.github.com/articles/using-pull-requests/).

## Authors
This code was created by Zane Fink (zanef2@illinois.edu), Konstantinos Parasyris (parasyris1@llnl.gov), 
Praneet Rathi (prathi3@illinois.edu), and Giorgis Georgakoudis
(georgakoudis1@llnl.gov), assisted with design input from Harshitha Menon (gopalakrishn1@llnl.gov).

## License
This repo is distributed under the terms of the Apache License (Version
2.0) with LLVM exceptions. Other software that is part of this
repository may be under a different license, documented by the file
LICENSE in its sub-directory.

All new contributions to this repo must be under the Apache License (Version 2.0) with LLVM exceptions.

See files [LICENSE](LICENSE) and [NOTICE](NOTICE) for more information.

SPDX License Identifier: "Apache-2.0 WITH LLVM-exception"

LLNL-CODE- 825539