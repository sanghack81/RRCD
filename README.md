# Towards Robust Relational Causal Discovery

Code for the UAI 2019 paper. RRCD uses pyRCDs for relational models and SDCIT
for conditional independence testing.

## Installation

Use Python 3.12 or newer. Install Graphviz and its development headers before
installing Python dependencies (`brew install graphviz` on macOS, or
`sudo apt-get install graphviz graphviz-dev` on Ubuntu).

SDCIT needs the external Blossom V source at build time. Its `blossom5/`
directory is not part of the GitHub repository, so installing SDCIT directly
from a Git URL does not work. The [Blossom V source and research-use license](https://pub.ista.ac.at/~vnk/software.html)
are published by its author; `./setup.sh` downloads version 2.05.

Run these commands from an RRCD checkout. The requirements install NumPy and
Cython before the native extensions are built:

```bash
python -m pip install -r requirements.txt
git clone https://github.com/sanghack81/SDCIT.git ../SDCIT
(cd ../SDCIT && ./setup.sh && python -m pip install --no-build-isolation -e .)
python -m pip install --no-build-isolation -e .
python -m pytest rrpcd/tests
```

The requirements file installs [`pyRCDs`](https://github.com/sanghack81/pyRCDs)
from GitHub. [`SDCIT`](https://github.com/sanghack81/SDCIT) must be built
separately as shown above.
