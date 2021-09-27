[Robust Training in High Dimensions via Block Coordinate Geometric Median Descent](https://arxiv.org/pdf/2106.08882.pdf)
=================================================================================
Geometric median (GM) is a classical method in statistics for achieving a robust estimation
of the uncorrupted data; under gross corruption, it achieves the optimal breakdown point of
0.5. However, its computational complexity makes it infeasible for robustifying stochastic
gradient descent (SGD) for high-dimensional optimization problems. In this paper, we show
that by applying Gm to only a judiciously chosen block of coordinates at a time and using
a memory mechanism, one can retain the breakdown point of 0.5 for smooth non-convex
problems, with non-asymptotic convergence rates comparable to the SGD with GM.

Citation  
------------
Kindly cite the following work:    
```
@article{acharya2021robust,
  title={Robust Training in High Dimensions via Block Coordinate Geometric Median Descent},
  author={Acharya, Anish and Hashemi, Abolfazl and Jain, Prateek and Sanghavi, Sujay and Dhillon, Inderjit S and Topcu, Ufuk},
  journal={arXiv preprint arXiv:2106.08882},
  year={2021}
}
```
