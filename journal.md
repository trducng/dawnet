# Incorporating the training and fitting of model into one session

|Start Date|End Date  |
|----------|----------|
|2018-09-09|2018-09---|

## Description

The goal of this project is to create handwriting OCR benchmarks to serve as comparision for models developed in the future. Specifically, this project recreate 4 models:

- Toni's vanilla model using (CNN -> BiLSTM -> CTC)
- Toni's improved version with attention and language modeling
- YOLO for object detection

The expected output:
- Have some comparision models
- Have a base model that we can play and tweak around for later improvements

*Tags*: benchmark

## Note

It is not possible to initialize a `torch.nn.DataParallel` inside a model class (`nn.Module`) because of internal infinite recursion between the `torch.nn.Dataparallel` and `nn.Module`. This happens since (1)  `torch.nn.DataParallel` is also a subclass of `nn.Module`, and (2) anytime a class subclassing `nn.Module` is initialized inside a `nn.Module`, it will be referred inside a parent `nn.Module` and (3) when you wrap a `nn.Module` model with `nn.DataParallel`, the `nn.DataParallel` contains `nn.Module`. Hence, the flow looks like this:

`Module A` -> contains `DataParallel of Module A` -> contains `Module A` -> contains `DataParallel of Module A` -> contains `Module A` -> ...

We would like to have training-related materials reside in the **agent**, the same way that a human's learning mechanism resides within the boundary of a person. The advantage of this formulation would be easier understanding of the involved components.

## Deliverables


## Interpretation


## Conclusion
