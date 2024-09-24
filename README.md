# KirasFM

**K**iras is an **I**terative, **R**obust, and **A**daptive **S**chwarz Solver **f**or **M**axwell

This repository contains the Maxwell Solver KirasFM, a framework built on top of deal.II that provides all necessary tools for solving time-harmonic Maxwell's equations with an optimized Schwarz method.

## Citation

Please use the *Cite this repository* button in the *About* section of this repository.

## Installation - Dependencies

The dependencies required are deal.II, Trilinos and boost.

### Obtaining deal.II via DCS2

This CMake script installs deal.II, along with its dependencies, including Trilinos. It should be the easiest way to install deal.II with all its dependencies. For more details, see <https://github.com/kinnewig/dcs2>.

tl;dr:

1. Step: Download dcs2:

```
git clone https://github.com/kinnewig/dcs2.git
cd dcs2
```

1. Step: Run the install script:

```
./dcs.sh  -b </path/to/build> -p </path/to/install>
```

If you have any problems, feel free to open an issue on <https://github.com/kinnewig/dcs2/issues>

### Obtaining deal.II via candi

candi is a bash script, to install deal.II, along with all its dependencies, including Trilinos, see <https://github.com/dealii/candi>.

## Usage

### Obtaining KirasFM

1. Download KirasFM

   ```
   git clone https://github.com/kinnewig/KirasFM.git
   ```


1. Compile KirasFM

   ```
   cd KirasFM
   cmake -S . -B build -D DEAL_II_DIR=</path/to/dealii>
   cmake --build build
   ```

### Run the example 

By default, a simplified waveguide is selected as an example. To run the example, specify the number of subdomains that shall be used in the options.prm file. Then run the program by:

```
mpirun -np <number of subdomains> ./build/KirasFM
```

  
Other geometries, for example, are available through the KirasFMGridGenrator classes.
