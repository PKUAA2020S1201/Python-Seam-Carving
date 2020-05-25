# Python-Seam-Carving

Python implementation of Seam Carving.

## Introduction

This is a python implementation of seam carving algorithm, with some improvement strategies.

For detailed information of these methods, please refer to `开题报告.md` or Google. 
Actaully, none of them is difficult and I believe you can totally understand them by reading the source code.

## How to Use

We placed the core code in a module named `sc2`, 
which means that you can just import it and build your own procedures conveniently.

Moreover, we released many Jupyter Notebooks (`.ipynb`) which we used in experiments.
These notebooks can help you understand the API.

Maybe a detailed API doc will come out soon.

## Speed Up

Python script is slower compared to C/C++, thus we used numba JIT to speed up a lot parts of calculation.
However, sometimes JIT would raise unexpected error, sometimes even return wrong results without warnings. 
So, we wraped `numba.jit` and created a new wrapper called `sc2.utils.just_in_time`.
You can use it to replace `numba.jit` and the advantage is that we created a global configuartation in `sc2.utils.jit`
and you can easily switch jit on/off using `sc2.utils.jit.enable()` and `sc2.utils.jit.disable()`.
This is espacially usefull when debuging.

Once, we implemented an energy function totally in Python, which means that we start from simulating convolution using loops.
As you can imagine, it works but it's too slow.
So we recommand OpenCV based energy functions, which are actually implemented in C/C++ and run extremely fast.

## Channel Order

We use OpenCV a lot, such as image reading and writing, image processing, image displaying and mask drawing.
OpenCV use BGR as the default channel order, while we prefer RGB more.
Thus we implemented our own functions to read and write images, 
and any image (np.ndarray) you get is in RGB order,
and any method you use receives and returns images in RGB order.

## Why This Exists

Actually, this repository is our team project for *算法设计与分析*.

## Credit and Copyright

*to be finished*
