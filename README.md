# Source code and images for: Driven particle flux through a membrane: Two-scale asymptotics of a diffusion equation with polynomial drift

This repository contains the source code, produced data and images for [this][1] paper.

## Requirements

 - Python 3
 - `numpy`
 - `matplotlib`
 - [FEniCS][2] with mshr

## Structure

The script `simulation.py` computes the finite-thickness upscaled membrane system described in Section 5 of the manuscript.
Run with
```bash
python3 simulation.py
```

The script `slanted_axis.py` computes the same model, this time with a specific diffusion tensor introducing a slant between the diffusion axes and the domain axes.
Run with 
```bash
python3 slanted_axis.py
```
The folder `images` contains all the images used in the numerical section of the manuscript.

The folder `results` contains a VTK file of a simulation that can be visualised using [Paraview][3] or a similar kind of visualisation program.

## Contact

Feel free direct any issues, questions, or comments to the owner of the repository.

[1]: https://arxiv.org/abs/1804.08392
[2]: https://fenicsproject.org/
[3]: https://www.paraview.org/
