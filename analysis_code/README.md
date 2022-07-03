# DNA Bendability Analysis

This repository contains various utility tools to analyze DNA Bendability. 

## Setup Repository 

**Use virtual environment** 

Python: 3.9+ required

```sh 
python3 -m venv venv            # Create venv. Only once.
source venv/bin/activate        # Activate venv. 
pip install -r requirements.txt # Install packages. Only once.
```

**Install Source Code** 

To run source code, you need to install it first. Install source code in virtual env as package in editable mode.  

```sh
cd src/
pip install -e .        # Do this everytime venv is activated
```

## Running Code 

Now, you are ready to run. Invoking functionalities is simple. In `src/main.py`, create proper objects and call it's public functions. As an example, for plotting average intrinsic cyclizability around nucleosome dyads in chromosome V, paste the following code in `main.py`. 

```py
from chromosome.chromosome import Chromosome
from chromosome.nucleosomes import Nucleosomes

if __name__ == '__main__':
    Nucleosomes(Chromosome('VL')).plot_c0_vs_dist_from_dyad_spread()
```

Then, run: `python3 main.py`. 

It should create a plot in `figures/nucleosome/` directory.

To initialize an object, invoke a function or understand the expected behavior, take a look at corresponding test files, docstrings and function annotations. 

Add new functionalities in a `Class` or in a module.

## Testing 

Tests are useful for for making sure that everything is working fine. Apart from running the whole test suite, you can run a single test module or selected tests.

```sh 
cd tests/
python3 -m pytest                          # Run whole test suite; ~30 mins
python3 -m pytest --testmon                # Run only tests affected by recent changes
python3 -m pytest util/test_reader.py      # Run a single test module
python3 -m pytest conformation/test_loops.py -k multichrm 
                                           # Run tests in test_loops.py module containing substring 'multichrm' 
```