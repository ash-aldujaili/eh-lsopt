# The EmbeddedHunter Algorithm for Large-Scale Black-Box/Derivative-Free Optimization :zap:

This repository hosts the code for the `EmbeddedHunter` algorithm for large-scale black-box optimization proposed in the **Embedded Bandits for Large-Scale Black-Box Optimization** [paper] (https://arxiv.org/abs/1611.08773), which is accepted at the thirty-first AAAI conference on artificial intelligence ([AAAI-17] (http://www.aaai.org/Conferences/AAAI/2017/aaai17call.php)).


# Contents

1.  `Embedded Bandits for Large-Scale Black-Box Optimization.ipynb` : a notebook for a quick demonstration.
2.  `run_aaai_demo.py`: a python script for running AAAI'17 paper experiments.
3.  `Benchmark.py`: a python module for the large-scale problems used.
4.  `Algorithms.py`: a python module implementing `RESOO`, `SRESOO`, and `EmbeddedHunter`.
5.  `*.pickle`: a collection of pickle files saving the outcome of `run_aaai_demo.py`
6.  `plot-table.tex`: a tex file to reproduce Figure 1 of the AAAI'17 paper.


To run all the experiments (it will take around 6 days), `cd` to the directory and execute the following
~~~
>>python run_aaai_demo.py -1
~~~
Since all the `*.pickle` files are there, the above command will just create a folder, which has all the experiments' results in a list of tabulated files. With these files at hand, you can now compile `plot-table.tex` to create the pdf version of AAAI'17 paper's Figure 1.



# Citation

If you write a scientific paper describing research that made use of this code, please consider citing the following paper:
~~~
@inproceedings{ash2017eh,
  title={Embedded Bandits for Large-Scale Black-Box Optimization},
  author={Abdullah Al-Dujaili and S. Suresh},
  booktitle={Thirty-First AAAI Conference on Artificial Intelligence (AAAI'17},
  year={2017}
}
~~~

