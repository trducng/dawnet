# Incorporating the training and fitting of model into one session

|Start Date|End Date  |
|----------|----------|
|2018-09-09|Delayed   |

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

----------------------------------------
# Visualization for convolutional layers

|Start Date|End Date  |
|----------|----------|
|2018-10-24|2018-10-31|

## Description

It is important to understand what happens to the data and the weights inside the neural network in order to make informed decision on how to improve the model. Otherwise, model improvement is basically trial-and-error, depends largely on luck.

Given a training model, it is good to know:
- What are the things that a specific feature map detects: the collection of images that excites that feature map, and in that collection, which parts particularly excite the feature map?
    + Is it possible to take a random image, and optimize that image on that feature map
- The learning progression of a feature map, from when it is randomized completely to when it is learned. Record the value of all feature maps on a group of images during training.

As a result, we need to:
- [x] Quickly point out original patch in the image that is responsible for a large activation of a neuron in feature map (python function on Jupyter notebook)
- [x] Point out the specific pattern in that original patch that makes the neuron activation in the feature map large. *How does the neuron activation changes when we change the pattern in that patch (python function on Jupyter notebook). Answering this question might require a gradient-based method to observe the change in activation neuron in response to change in input image.*
- [x] Have a function to cut a portion of an image and paste that portion to another image and see activation. (can do python function with script, needs to quickly get the pixel indices)
- [x] Have a function to quickly modify (e.g. draw into that image), and augment that image (e.g. add noise, blur,...), allow to view the feature maps in real time when this happens. (the augmentation can be done with python function, the draw seems to need some kind of interactive Python/Jupyter Notebook functionality)
- [x] Have a function to visualize feature activations, weights and biases.
- [] A training procedure to optimize a given input image to satisfy a result (python function on Jupyter notebook) -> Nguyen .et al mechanism

**Roadmap**:
+ get the indices of input region that is responsible for a neuron activation
    + given a channel index a layer (a specific 2D location might not be provided, if that is the case, find for all activated indices), collect the images and the specific patches that make that channel activated
+ widgets to view the effects of changes in the input images (or much better, changes in one layer of a model)
- implement other visualization techniques in the literature -> longer term.
    + https://distill.pub/2017/feature-visualization/
    + Google Deep dream
    + http://cs231n.github.io/understanding-cnn/
    + https://github.com/utkuozbulak/pytorch-cnn-visualizations
    + https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks
    + http://yosinski.com/deepvis
    + https://jacobgil.github.io/deeplearning/filter-visualizations

## Discussion

## Deliverables

## Interpretation


----------------------------------------
# Keep track of model's number of floating point operations

|Start Date|End Date  |
|----------|----------|
|2018-11-10|2018-11-12|

## Description

In developing a model, it is sometimes important to balance accuracy performance with computing performance. If a model achieves 95% accuracy at a cost of 5 minutes, while another model achieves 93% accuracy at a cost of 3 seconds, then the second model can become appealing in several use cases (for example, on mobile...). To track a model's computing performance, people usually use the models number of parameters and the model's number of floating point operations. The former one is easy to calculate, as we only need to count the total number of elements in parameter matrices. The later one is a little bit trickier, as Pytorch does not out-of-the-box support this calculation, and different layers have different formula to calculate the number of floating point operations.

On github, there are 2 sources integrate this kind of calculation (the implementations look largely similar):
- https://github.com/ShichenLiu/CondenseNet/blob/master/utils.py
- https://github.com/sovrasov/flops-counter.pytorch/blob/master/flops_counter.py

The second link seems to be more holistic in its calculation. It lacks RNN floating point calculation, which we will need to fill in.

Currently, I am thinking whether:
- integrate the calculation directly by subclassing,
- use mixings
- integrate the calculation as hooks (as in the above implementations)

|                    |Pros|Cons|
|--------------------|----|----|
| Directly integrate |(1) all the calculations can be done directly, (2) might work with other kinds of integrations in this module, (3) can be called once in the model|(1) will require a lot lot lot of subclassing (2) as a result, the code can be easily messy, (3) will have hard time when Pytorch interface changes, (4) not portable|
|Mixings             |(1) can be reused for similar layers| (1) cognitive cost to remember to add mixins to layers in the model, (2) use not be portable to non dawnet models, (3) can have hard time organizing the mixins|
|As hooks            |(1) reusable accross Pytorch models, (2) can be called once in the model|(1) requires a lot of if-else inside, (2) can have hard time organizing the hooks, (3) rather nonPythonic to access hidden methods|