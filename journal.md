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

## Deliverables

- [x] `diagnose/statistics.py`: obtaining model statistics
- [x] `diagnose/trace.py`: trace activation information backward


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

The 3rd option is a better choice for initial implementation:
- code already have
- works a lot of place
- easy integration

After this integration, thinks about how to incorporate to _Base_Model

## Deliverables

- [x] initial integration of hooks flop counting
- [x] add rnn operation

## Conclusion

Learn a little more about LSTM. The main gotcha is the number of hidden neurons is equal to the number of cells.


----------------------------------------
# Incorporate Lucid's feature visualization

|Start Date|End Date  |
|----------|----------|
|2018-11-17|2018-11-19|

## Description
Lucid's feature visualization [1](https://distill.pub/2017/feature-visualization/) [2](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/render.py) [3](https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/tutorial.ipynb) provides fascinating way to visualize the inner working of a hidden neuron or a combination of hidden neurons in the neural network. If any `dawn` model has this capability, we can inspect model performance, and (I get an intuition) we can learn more on how to improve model. Utilitarian reasons aside, implmementing this functionality would be fascinating as well. Let's see how it is going.

# Experiments
The heart of Lucid's visualization seems to reside in this [script](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/render.py), especially in the function `make_vis_T`. This function requires:
- a trained model
- a combination of neurons (here called objectives) to optimize (the nice thing is the objective can be combined from multiple separate objectives)
- some preconditioned image parameterization, by default: RGB 128x128, spatial Fourier transformation, decorrelated color (knowing some color components cannot predict remaining color components)
- an optimization, by default Adam w/ learning rate 0.05, for 512 iterations
- some image preprocessing transformations, by default: random padding -> random crop (jitter) 8 -> random pixel scaling -> random rotation -> random crop (jitter) 4

Then, in order to obtain the image, they maximize the output activation of:
- a single neuron       -> currently the simplest thing to implement
- a single channel
- some direction
- single (x, y) position along a direction
- visualize according to cosine similarity
- deep dream method
- a combination of channels (some interpolation)

...(suspend)


----------------------------------------
# Incorporate super-convergence for training

|Start Date|End Date  |
|----------|----------|
|2018-12-05|2018-12-08|

## Description

Super-convergence seems to be much hyped. It is inspired from the observation that large learning rate produces short term negative effect and but provides long term beneficial effect. The author of this method suggests to triangulate the learning rate between maximum and minimum bounds of learning rate. It is also claimed by the author that super-convergence helps when we have limited amount of training data.

It is referred in these papers:
- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
- [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)

From the first paper, we need to pay attention to some points:
- Terms: max_lr, base_lr, stepsize
- Visual shape: below is triangular variation, we can have hypobolic or other shape, but the paper claims that they all have similar performance, hence triangular variation is preferred due to simplicity

```
_________________________________________max_lr
        /\          /\          /\           
       /  \        /  \        /  \
      /    \      /    \      /    \
     /      \    /      \    /      \
    /        \  /        \  /        \
___/__________\/__________\/__________\__base_lr
  |<---->| stepsize
```
- The `max_lr`, `base_lr` value can be determined in 2 ways:
    + Increase the learning rate gradually, choose the learning rate inducing increasing accuracy as `base_lr`, choose the learning rate marking the degradation as `max_lr`
    + If you know the converging learning rate, set `max_lr` = 2 * that learning rate, and `base_lr` = `max_lr` / 4
- The `stepsize` value is 2 to 10 times the epoch. Preliminary experiments confirm this experiment, for a training size of 80 000, batch size 2, the training iteration 90 000 (actually 1st epoch) provides much better prediction, than model of earlier iteration.

The second paper (super-convergence) provides:
- The `max_lr`, `base_lr` value can be determined with increasing the learning rate gradually, choose the learning rate marking degradation as `max_lr`, and divide this number by 3 or 4 to obtain the `base_lr`
- Use LR range test to see if super-convergence is possible for an architecture (LR range is essentially the above bullet point)
- Before stopping training, allow the learning rate to shrink several magnitude smaller than the `base_lr`

```
                                         max_lr
        /\          /\          /\           
       /  \        /  \        /  \
      /    \      /    \      /    \
     /      \    /      \    /      \
    /        \  /        \  /        \
   /          \/          \/          \  base_lr
  |<---->| stepsize                    \
                                        \
                                         \
                                          \
                                           \
                                            \
                                             \
                                              \
                                       |<---->| stepsize
```

Variation:

```
                                                           max_lr
        /\                /\                /\           
       /  \              /  \              /  \
      /    \            /    \            /    \
     /      \          /      \          /      \
    /        \        /        \        /        \
   /          \______/          \______/          \______  base_lr
  |<---->| stepsize |                 |                 | -> save checkpoint
```

In order to create super-convergence, we need: `max_lr`, `base_lr`, `stepsize`, `current_iteration`, and a signal to know when training needs to stabelize, and when training is stabelize, how to handle learning rate. In the mean time, this can be accomplished by manual intervention, and the learning rate is reduced stepwise (or with other traditional learning rate scheduler methods).

A learning rate finder function would require:
- min_lr
- max_lr
- decay_steps
- num_iterations
- model that has `x_learn` and `x_evaluate`
- optimizer object inside model (to change the learning rate)
- data generator
- higher_better


## Deliverables

- [x] `training/hyper.py:SuperConvergence`: super-convergence training
- [x] `diagnose/vis.py:draw_history`: visualize learning rate (requires combining multiple plot, requires set appropriate default height and width)
- [x] `training/hyper.py:SuperConvergence`, `models/perceive.py:BaseModel`: save the scheduler into checkpoint
- [x] `training/hyper.py:SuperConvergence`, `models/perceive.py:BaseModel`: save seperate checkpoints when the learning rate is at the lowest
- [x] `models/base.py:batch_infer`: ensemble inference
- [-] ~~validation loss and accuracy during training in order to save best model, these values should also be contained inside the history progress~~ (this requirement is too specific to incorporate into BaseModel - the numbers and kinds of metrics to monitored change depending on project)
- [x] `training.hyper.py:lr_finder`: learning rate finder
- [x] `models/perceive.py:DataParallel`: DataParallel - automatically add callable functions and variables of `self.module` to `DataParallel
