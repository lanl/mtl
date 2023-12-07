# Description
**Multi-task learning and inference from seismic images**

The machine learning model infers and refines a denoised, higher-resolution image (DHR for short), a relative geological time (RGT for short) volume and geological fault attributes (including location, dip, strike; Fault for short) from a noisy, low-resolution seismic image. The code implements the training, validation, and test for two functionalities: (i) multi-task inference (MTI), which infers DHR, RGT and Fault, and (ii) multi-task refinement (MTR), which refines DHR, RGT and Fault output from MTI. 

The work was supported by an FY23 Rapid Response project (GRR3KGAO) of Center for Space and Earth Science (CSES), Los Alamos National Laboratory (LANL). LANL is operated by Triad National Security, LLC, for the National Nuclear Security Administration (NNSA) of the U.S. Department of Energy (DOE) under Contract No. 89233218CNA000001. The research used high-performance computing resources provided by LANL's Institutional Computing program. 

LANL open source approval reference O4656.

# Reference
LA-UR-23-20649: Gao, 2023, Iterative multi-task learning and inference from seismic images, [Geophysical Journal International](https://doi.org/10.1093/gji/ggad424)

# Requirement
The code is implemented with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/). To train/valid/test the model, please

```
pip install torch torchvision torchaudio torchmetrics tensorboard lightning
```
Other packages may be needed depending on your Python distribution and installed packages. 

# Use
```
cd train; ruby train2.rb; ruby train3.rb
cd test; ruby test2.rb; ruby test3.rb
```

For large 3D images, GPU may run out of memory. In such a case, you can use CPU by setting ```--gpus_per_node=0``` for inference/refinement, or decompose the input image into blocks and merge results. 

Only example training/validation/test data are provided due to the size limit. Full training/validation data and trained models may be released in the future.

# License
This program is Open-Source under the BSD-3 License.

&copy; 2023. Triad National Security, LLC. All rights reserved. 

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  
# Author
Kai Gao, <kaigao@lanl.gov>
