# Description
**Multi-task learning and inference from seismic images**

The machine learning model infers and refines a denoised, higher-resolution image (DHR for short), a relative geological time (RGT for short) volume and geological fault attributes (including location, dip, strike; Fault for short) from a noisy, low-resolution seismic image. The code implements the training, validation, and test for two functionalities: (i) multi-task inference (MTI), which infers DHR, RGT and Fault, and (ii) multi-task refinement (MTR), which refines DHR, RGT and Fault output from MTI. 

The work was supported by an FY23 Rapid Response project (GRR3KGAO) of Center for Space and Earth Science (CSES), Los Alamos National Laboratory (LANL). LANL is operated by Triad National Security, LLC, for the National Nuclear Security Administration (NNSA) of the U.S. Department of Energy (DOE) under Contract No. 89233218CNA000001. The research used high-performance computing resources provided by LANL's Institutional Computing program. 

# Reference
LA-UR-23-20649: Gao, 2023, Iterative multi-task learning and inference from seismic images. 

# Requirement
The code is implemented with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/). To train/valid/test the model, please

```
pip install torch torchvision torchaudio torchmetrics tensorboard lightning
```
Other packages may be needed depending your Python distribution. 

# Use
```
cd train; ruby train2.rb; ruby train3.rb
cd test; ruby test2.rb; ruby test3.rb
```

For large 3D images, GPU may run out of memory. In such a case, you can use CPU by setting ```--gpus_per_node=0``` for inference/refinement, or decompose the input image into blocks and merge results. 

Only example training/validation/test data are provided due to the size limit. Full training/validation data and trained models may be released in the future under separate approval. 

# License
LANL open source approval reference O4656.

&copy; 2023. Triad National Security, LLC. All rights reserved. 

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
  
# Author
Kai Gao, <kaigao@lanl.gov>
