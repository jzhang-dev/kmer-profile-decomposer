# *k*-mer Profile Decomposer

Model a given k-mer profile as a mixture of multiple evenly spaced negative binomial distributions, and fit parameters by minimizing the residual sum of squares errors. 

The model used here is a generalized version of the [GenomeScope](https://github.com/schatzlab/genomescope) model, extended to allow explicit modelling of errors and fitting an arbitrary number of peaks. 

## Installation

```sh
pip install git+https://github.com/jzhang-dev/kmer-profile-decomposer
```

## Usage

See [demo.py.ipynb](https://github.com/jzhang-dev/kmer-profile-decomposer/blob/main/demo.py.ipynb).



## References

Vurture, G. W., Sedlazeck, F. J., Nattestad, M., Underwood, C. J., Fang, H., Gurtowski, J., & Schatz, M. C. (n.d.). GenomeScope: Fast reference-free genome profiling from short reads. 3.
