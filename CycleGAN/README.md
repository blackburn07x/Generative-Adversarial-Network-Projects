# CycleGAN Implementation for Horse-to-Zebra Translation using Tensorflow 2.0


## Table of Contents

- [Overview](#overview)
- [CycleGAN](#cyclegan)
- [Requirements](#requirements)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [Conclusion](#conclusion)

## Overview

This project is an implementation of the CycleGAN (Cycle-Consistent Adversarial Networks) for the task of translating horse images into zebra images. CycleGAN is a powerful generative adversarial network architecture that excels in image-to-image translation tasks, where no paired training data is required. Our implementation showcases its ability to seamlessly transform horse images into zebra-style images.

## CycleGAN

CycleGAN is a deep learning model known for its unpaired image-to-image translation capabilities. The key features of CycleGAN include:

- Unpaired Translation: CycleGAN can translate images from one domain to another without the need for paired training data, making it incredibly versatile.

- Cycle-Consistency Loss: It enforces cycle-consistency between the generated images, ensuring that the translated image can be effectively translated back to the original domain.

- Adversarial Learning: By using a combination of generator and discriminator networks, CycleGAN achieves realistic image translation results.

## Requirements

- Python 3.7 or later
- TensorFlow 2.0 or higher
- GPU for faster training (recommended)

## Acknowledgments

The implementation of CycleGAN draws inspiration from the original work by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros, as described in their research paper.

## Results

![Horse-to-Zebra Translation](results/picture1.png)<br>
*Horse Image (Left) Translated to Zebra Image (Right)*

![Horse-to-Zebra Translation](results/picture2.png)<br>
*Horse Image (Left) Translated to Zebra Image (Right)*

![Horse-to-Zebra Translation](results/picture3.png)<br>
*Horse Image (Left) Translated to Zebra Image (Right)*

Our CycleGAN model successfully translates horse images into zebra-style images, showcasing the power of unpaired image translation. The generated images maintain the visual fidelity of zebras while preserving the essence of the original horse images.

## References

- [CycleGAN Research Paper](https://arxiv.org/abs/1703.10593)

