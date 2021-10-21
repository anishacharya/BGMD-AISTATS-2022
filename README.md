[Robust Training in High Dimensions via Block Coordinate Geometric Median Descent](https://arxiv.org/pdf/2106.08882.pdf)
=================================================================================
Anish Acharya, Abolfazl Hashemi, Prateek Jain, Sujay Sanghavi, Inderjit Dhillon, Ufuk Topcu.

Abstract
------------
Geometric median (GM) is a classical method in statistics for achieving a robust estimation
of the uncorrupted data; under gross corruption, it achieves the optimal breakdown point of
0.5. However, its computational complexity makes it infeasible for robustifying stochastic
gradient descent (SGD) for high-dimensional optimization problems. In this paper, we show
that by applying Gm to only a judiciously chosen block of coordinates at a time and using
a memory mechanism, one can retain the breakdown point of 0.5 for smooth non-convex
problems, with non-asymptotic convergence rates comparable to the SGD with GM.

![](https://github.com/anishacharya/BGMD/blob/main/readme_utils/bgmd_algo.png)

![](https://github.com/anishacharya/BGMD/blob/main/readme_utils/bgmd_fig1.png)

![](https://github.com/anishacharya/BGMD/blob/main/readme_utils/bgmd_theoru.png)


Citation  
------------
If you find the algorithm useful for your research consider citing the following article:     
```
@article{acharya2021robust,
  title={Robust Training in High Dimensions via Block Coordinate Geometric Median Descent},
  author={Acharya, Anish and Hashemi, Abolfazl and Jain, Prateek and Sanghavi, Sujay and Dhillon, Inderjit S and Topcu, Ufuk},
  journal={arXiv preprint arXiv:2106.08882},
  year={2021}
}
```

If you find the code useful for your research consider giving a :star2: and citing the following:
```
@software{Acharya_BGMD_2021,author = {Acharya, Anish},doi = {10.5281/zenodo.1234},month = {10},title = {{BGMD}},url = {https://github.com/anishacharya/BGMD},version = {1.0.0},year = {2021}}
```
