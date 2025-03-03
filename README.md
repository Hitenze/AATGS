# AATGS: Anderson Acceleration with Truncated Gram-Schmidt

Non-official research code implementing the AATGS algorithm. This code was used for development and testing and was not used to generate the figures in the associated paper.

## Installation

### Configure
```bash
./configure
```

### Configure with options
```bash
# Use OpenBLAS
./configure --with-openblas

# Use MKL with single precision
./configure --with-mkl --with-f32
```

### Optional
Manually modify `Makefile.in` if needed.

### Build
```bash
make
```

## Reference

**Anderson Acceleration with Truncated Gram-Schmidt**  
Z. Tang, T. Xu, H. He, Y. Saad, Y. Xi  
*SIAM Journal on Matrix Analysis and Applications* (2024)  
DOI: [10.1137/24M1648600](https://doi.org/10.1137/24M1648600)

## Contact

Questions? Email Tianshi Xu (txu41@emory.edu).