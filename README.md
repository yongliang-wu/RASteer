# RASteer

Retain Aware Activation Steering for Concept Erasure in Diffusion Models

Yongliang Wu, Haori Lu, Yulun Wu, Jinqi Luo, Xingyu Zhu, Yaoyao Liu

[Project page](https://rasteer.cvmlgroup.web.illinois.edu/)

## Method

RASteer removes unwanted concepts from diffusion models through activation steering without training or changing model weights. It builds a subspace for concepts to preserve, removes shared components from the erasure direction, and adaptively calibrates this removal at each layer and denoising step.

## Results

Experiments on unsafe content, instances, and artistic styles across multiple diffusion backbones show improved concept preservation while maintaining effective erasure. On the I2P benchmark, RASteer reduces exposed body part detections from 612 to 13.

## Code

Code coming soon.
