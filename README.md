## Subclass deep neural networks

This repository contains the code for the paper Subclass deep neural networks: re-enabling neglected classes in deep network training for multimedia classification.

## Introduction

We propose a new criterion for identifying the so-called "neglected" classes during the training of DNNs, i.e. the classes which stop to optimize early in the training procedure. Moreover, based on this criterion a novel cost function is introduced, that extends the cross-entropy loss using subclass partitions for boosting the generalization performance of the neglected classes. The proposed approach has been used to extend the VGG16 and Wide Residual Network architectures, and has been validated in CIFAR-10, CIFAR-100, SVHN and on the large-scale YouTube-8M video dataset.

## Dependencies

To run the code use Pytorch 1.1.0 and Tensorflow 1.13.1 or later.

## Usage

To run the code for the different datasets and networks use the corresponding settings described in the paper. For instance, for cifar10, cifar100, and svhn with vgg16 use:

python notebook.py --dataset cifar10 --model vgg16 --data_augmentation --cutout --length 16 --subclass --numClasses2Repartition 2

python notebook.py --dataset cifar100 --model vgg16 --data_augmentation --cutout --length 8 --subclass --numClasses2Repartition 10

python notebook.py --dataset svhn --model vgg16 --cutout --length 20 --subclass --numClasses2Repartition 2

## License and Citation

This code is provided for academic, non-commercial use only. If you find this code useful in your work, please cite the following publication where this approach is described:

N. Gkalelis, V. Mezaris, "Subclass deep neural networks: re-enabling neglected classes in deep network training for multimedia classification", Proc. 26th Int. Conf. on Multimedia Modeling (MMM2020), Daejeon, Korea, Jan. 2020.

Bibtex:
```
@INPROCEEDINGS{SDNN_MMM2020,
               AUTHOR    = "N. Gkalelis and V. Mezaris",
               TITLE     = "Subclass deep neural networks: re-enabling neglected classes in deep network training for multimedia classification",
               BOOKTITLE = "Proc. 26th Int. Conf. on Multimedia Modeling (MMM2020)",
               ADDRESS   = "Daejeon, Korea",
               MONTH     = "Jan.",
               YEAR      = "2020"
}
```

## Acknowledgements

This work was supported by the EUs Horizon 2020 research and innovation programme under grant agreement H2020-780656 ReTV.


