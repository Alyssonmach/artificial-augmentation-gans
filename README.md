# Generation of Artificial Images for Data Augmentation Using Generative Adversarial Networks

This project aims to generate new images using machine learning techniques called generative adversarial networks. These techniques allow the generation of images that appear real, but were created artificially.

This project is being developed as part of scientific initiation research (PIBIC) at the Federal University of Campina Grande, with the aim of applying these techniques to increase the amount of data available for training computer vision models.

### How it works

Generative adversarial networks are composed of two neural networks trained together. One of the networks, called a generator, is trained to create new images from random data. The other network, called the discriminator, is trained to identify whether an image is real or generated by the generator network.

During training, the two networks work together to improve the generator's ability to create images that look real and to improve the discriminator's ability to identify generated images. This allows the generator to create new images that are very similar to the real images.

### Project Reports

* [Project execution report UFCG]()

### Results

* [Data Augmentation Test Results](data-augmentation-experiments/planilha-resultados.pdf)

### Citation

```
@software{Machado_Geracao_de_Imagens_2023,
author = {Machado, Alysson and Veloso, Luciana and Araújo, Leo},
month = sep,
title = {{Geração de Imagens Artificiais para Aumento de Dados Utilizando Redes Adversárias Generativas}},
url = {https://github.com/Alyssonmach/artificial-augmentation-gans},
version = {1.0.0},
year = {2023}
}
```
