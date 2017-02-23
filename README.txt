A parallelized motion detection implementation of Yi et al.’s dual-mode SGM background subtraction model. 

If you use our implementation please cite:

@ARTICLE{2017arXiv170205156H,
   author = {{Henderson}, P. and {Vertescher}, M.},
    title = "{An Analysis of Parallelized Motion Masking Using Dual-Mode Single Gaussian Models}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1702.05156},
 primaryClass = "cs.CV",
 keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Distributed, Parallel, and Cluster Computing},
     year = 2017,
    month = feb,
   adsurl = {http://adsabs.harvard.edu/abs/2017arXiv170205156H},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


Main Paper Citation Dual-Mode Single Gaussian Models:

@inproceedings{yi2013detection,
  title={Detection of Moving Objects with Non-stationary Cameras in 5.8 ms: Bringing Motion Detection to Your Mobile Device},
  author={Yi, Kwang Moo and Yun, Kimin and Kim, Soo Wan and Chang, Hyung Jin and Choi, Jin Young},
  booktitle={Computer Vision and Pattern Recognition Workshops (CVPRW), 2013 IEEE Conference on},
  pages={27--34},
  year={2013},
  organization={IEEE}
}


The CUDA implementation of our code is in the cuda folder. Both the serial and parallelized tbb versions of our code reside in the same folder: serial_and_tbb. They can both be seen in the main.cpp and DualSGM.cpp files.


