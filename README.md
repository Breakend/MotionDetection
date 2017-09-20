A parallelized motion detection implementation of <a href="http://ieeexplore.ieee.org/document/6595847/">Yi et al.’s dual-mode SGM background subtraction model</a>. 

If you use our implementation please cite:

```
@article{henderson2017analysis,
  title={An Analysis of Parallelized Motion Masking Using Dual-Mode Single Gaussian Models},
  author={Henderson, Peter and Vertescher, Matthew},
  journal={arXiv preprint arXiv:1702.05156},
  year={2017}
}
```

The CUDA implementation of our code is in the cuda folder. Both the serial and parallelized tbb versions of our code reside in the same folder: serial_and_tbb. They can both be seen in the main.cpp and DualSGM.cpp files.


