# Adversarial Random Forests
Code and materials to reproduce paper "Adversarial Random Forests for Density Estimation and Generative Modelling", to appear in *Proceedings of the 26th International Conference on Artificial Intelligence and Statistics*. Preprint: https://arxiv.org/abs/2205.09435.

The proposed method is implemented in the `arf` R package, available on [CRAN](https://cran.r-project.org/package=arf) and [Github](https://github.com/bips-hb/arf). To install the package from CRAN, run: 
```R
install.packages("arf")
```
To install the development version from GitHub using `devtools`, run:
```R
devtools::install_github("bips-hb/arf")
```

Directories included in this repository:

* simulation: Directory containing code to reproduce visual examples and comparison with other tree-based approaches (Sec. 5.1)
* density_benchmark: Directory containing code to reproduce comparison with alternative PCs in density estimation (Sec. 5.2)
* generative_benchmark: Directory containing code to reproduce comparison with deep learning approaches in generative modeling (Sec. 5.2-5.3)
* appx_mnist: Directory containing code to reproduce comparison with conditional GAN for MNIST28 (Appx. B.5)

