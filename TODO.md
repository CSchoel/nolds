# TODOs

This file contains a list of general ideas for the future that are not yet detailed or important enough to be issues.

* Research expected values of DFA and sampen for lorenz system and add them to example
* Open up nolds for multidimensional input (check if Deniz Eroglu is right about skipping the delay embedding step that uses Takens' theorem)
    * Test multidimensional versions of algorithms with Lorenz system
* Implement algorithm to find "linear part" in a time series based on maximum subarray algorithm (~= "maximum linear subsequence")

## Roadmap for version 1.0

* complete Lorenz example
* use Lorenz example for unit tests
* multidimensional data + Lorenz test
* update README.rst with new algorithms
* add type hints
* reduce size of unit tests
* switch to pep 517 build system?
