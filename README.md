# Robust Training in High Dimensions via Block Coordinate Geometric Median Descent

Geometric median (GM) is a classical method in statistics for achieving a robust estimation
of the uncorrupted data; under gross corruption, it achieves the optimal breakdown point of
0.5. However, its computational complexity makes it infeasible for robustifying stochastic
gradient descent (SGD) for high-dimensional optimization problems. In this paper, we show
that by applying Gm to only a judiciously chosen block of coordinates at a time and using
a memory mechanism, one can retain the breakdown point of 0.5 for smooth non-convex
problems, with non-asymptotic convergence rates comparable to the SGD with GM.
