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

After inquiring "A Disciplined Approach to Neural Network Hyper-Parameters" from Leslie Smith, I acknowledge that Super-Convergence is a special case of cyclical learning rate, where there is only 1 cycle, and the maximum learning rate is really large, after that the learning rate is set exponentially smaller until the loss does not have any meaningful improvement. 1-cycle learning means that if we train the model for X iterations, we set the stepsize to be X/3, so that 2X/3 iterations will train the model cyclically, and in the other X/3 iterations, the learning rate will decrease exponentially.

## Deliverables

- [x] `training/hyper.py:SuperConvergence`: super-convergence training
- [x] `diagnose/vis.py:draw_history`: visualize learning rate (requires combining multiple plot, requires set appropriate default height and width)
- [x] `training/hyper.py:SuperConvergence`, `models/perceive.py:BaseModel`: save the scheduler into checkpoint
- [x] `training/hyper.py:SuperConvergence`, `models/perceive.py:BaseModel`: save seperate checkpoints when the learning rate is at the lowest
- [x] `models/base.py:batch_infer`: ensemble inference
- [-] ~~validation loss and accuracy during training in order to save best model, these values should also be contained inside the history progress~~ (this requirement is too specific to incorporate into BaseModel - the numbers and kinds of metrics to monitored change depending on project)
- [x] `training.hyper.py:lr_finder`: learning rate finder
- [x] `models/perceive.py:DataParallel`: DataParallel - automatically add callable functions and variables of `self.module` to `DataParallel


----------------------------------------
# Attack model by augmenting input images

|Start Date|End Date  |
|----------|----------|
|2018-12-11|2018-12-13|

## Description

A trained model might not be robust to generalize to test images. A common problem is that augmentation variations (more skew, more padding, more noises...) can influence model's prediction. As a result, it would be nice to instantly see the effect of data augmentation on model prediction. There are lots of variations that we can add to an image. In my experience, padding weirdly influences model's prediction, since padding does not significantly alter the visual characteristic of images (e.g. the prediction of image with white background should not be influenced by white padding but it does!).

In order to examine the effect of augmentation on the models, it is necessary to view the model channels' response to both the original image and to the augmented images. As a result, for a given image and an augmented image, the system must:
- show channels response to original image on the left side
- show channels response to augmented image on the middle side
- show the difference between channel (1) and channel (2)
(the three above points should be in 1 line of the image, the system should show multiple lines for multiple channels)

### Bugs found
- `run_partial_model`, `get_number_layers`, `get_layer` and `get_layer_indices` assume different layer indices:
    + `run_partial_model`, `get_layer`, and `get_layer_indices` assume that the index values include non-pytorch classes
    + `get_number_layers` does not include the indices inside 

## Discussion

It is commonly known that convolutional layers with max pooling provide translation equivariance. However, when viewing higher feature maps of a model using two images slightly different from each other (2 images have white background, one has 1 pixel of white padding than the other), it shows that the 2 feature maps are quite difference. It is shown below (left - feature1, middle - feature2, right - feature1-feature2):


![Feature maps difference](media/plots/diff_feature_maps.png?raw=true "Feature maps difference")

This pokes 2 questions:
- Why such drastic difference in feature map representations for 2 input images that visually indistinguishable?
- If we know that 2 very similar images provide this difference in feature map, what can we do to fix it?

To explore, we build a mechanism to see model output if we interfere with intermediate activations (done, using both `run_partial_model` and `predict_partial_model`). All observations below refer to the padding example above:
- if we take the max (`torch.max`) of 2 corresponding feature maps, the prediction modify slightly, but seems to become a little more unstable
- randomly pick a random activation value of a ReLU layer, and take a *max* also seems to increase the accuracy. This technique in some sense resembles dropout, in that instead of randomly dropping activation value to *min* (0), we randomly increase it to *max*

## Deliverables

- [x] `diagnose/attack.py:compare_model_response`: compare model input feature maps of 2 different input values
- [x] `diagnose/trace.py:predict_partial_model`: run the prediction from a feature map
- [x] fix the discussed bug -> basically a layer or any layer_idx considers valid layers only (those excluding `Sequential`...)




* Study the effect of initialization in trainings
* Understand more about convolutions kernel: convolutional kernels in DNN attempt to aggregate information around a local points, hence, the shape of the kernel defines what information will be taken into account inside the calculation
> ...convolution as a kind of information aggregation, but an aggregation from the perspective of a specific point: the point at which the convolution is being evaluated...  The output point wants to summarize information about the value of f(x) across the function’s full domain, but it wants to do so according to some specific rule
* What about flipping the weights. Usually, convolutional weights assume spatial dependency, what if we flip the position of convolutional weights?
* Follow the idea from here: https://arxiv.org/abs/1712.09913, what is the distance between the initial state of the model to the converged state, is there anyway to save the trajectory of the model during its training? The idea from the article allow for very minimal exploration of the loss landscape within a small area in either 1 or 2 dimension


----------------------------------------
# Visualize loss landscape

|Start Date|End Date  |
|----------|----------|
|2019-01-06|2019-01-08|

## Description

Visualizing the loss landscape is a fun way to partly understand what happens under the hood of a neural network. It can provide useful knowledge to know whether a model architecture is good for a particular dataset, or whether another choice of hyperparameters can provide better generalization power to the trained model. There are many methods to visualize or to peak into the inner-working of the model, this project will talk first about how to visualize loss landscape, inspired by the paper **Visualizing Loss Landscape of Neural Nets**.

This method is useful to:
- visualize the loss landscape of a model around a given parameter point (2D direction);
- visualize the loss landscape of a model from 1 parameter point to another parameter point (1D direction).

From the loss landscape visualization, one can figure out if one's model architecture and training setting are suitable.

This visualization method is inspired by **1-Dimensional Linear Interpolation** and **Contour Plots & Random Directions**. It differs in how to determine the direction length: it normalizes the direction length to be equal to the norm of the model weights, rather than using random direction length like in the later 2 methods. The normalization step avoids occurences where the same pertubations to large weights is smaller than to small weights, making incompatible comparision.

The basic idea of this visualization method is as follow. Given a trained weights and biases parameters \thetha_1 (which is a vector), pick a target point in the same parameter space \theta_2, such that the norm of (\delta = \theta_2 - \theta_1) is equal to the norm of \theta_1. Then for a sequence of \alpha in range of 0, and 1, calculate the loss value of model at weight \theta_1 + \alpha\delta.

Keep in mind:
- Saliency map
- Maximize neurons
- Maximize the output neurons
- For a given image, follow the most activated path: http://people.csail.mit.edu/torralba/research/drawCNN/drawNet.html
- For a trained model, pass through several images, and see which images ignite certain neuron
- RNN: Andrej Kaparthy, Seq2seq-Vis for machine translation
- DQNViz (VAST 2018)
- Visualize data:
    + tSNE
    + UMAP
    + Isomap
    + Sammon mapping
    + Multidimensional scaling 
- https://projector.tensorflow.org/
- Visualizing Deep Network Training Trajectories

Observation with CTC loss:
- the loss is unstable, moving from -1 to 0 to 1 along a normed distance result in a transition from chaos to sensible to chaos
- none of the observing point along the way provides the correct answer
- as a result, it is not possible to use weight jitter testing using this method for sequence problem.


----------------------------------------
# Make Agent's ability to use data flexibly

|Start Date|End Date  |
|----------|----------|
|2020-07-25|2020-07-26|

## Description

Reproducibility requires model to be run on a wide range of datasets, in different scenarios, during training and/or evaluation. The current model packaging method typically bundles the model with DataLoader, that is hard-coded to run on a specific dataset. If you want to run the model on a randomly-selected data, you have to go through the dataset definintion phase, hacking your way until you process that random data enough into a format that can be consumed by the model.

I believe much of that process can be automated if we rethink the role of data and incorporate it smartly to the training and evaluation process.

Target:
- data can be batched for training and inference (to get fast data)
 + still use the same `Dataset` and `DataLoader` models
- all observation can be augmented:
 + the augmentation function is declared inside the model
- user can run prediction for random data with minimal effort
 + add the `is_valid_action` and `is_valid_observation` functionalities
 + the observation modification step is handled inside Agent
 + the Dataloader is dynamically constructed inside Agent
- data formulation is model-agnostic.
 + the `Environment` class is supposed to be model agnostic.
 + data augmentation, input/output manipulation, batching of data is declared inside agent
 + TODO: might need a mechanism to automatically add those augmentation... from environment

Mentor is the ones who do the evaluation, using information from:
- Environment: can switch between environment
- Agent: expose the method to make prediction

Should be tested and applied to several projects before being sure that it can work in intelligence researching in general.

System design:
- The environment contains:
 + `Dataset`
 + `is_valid_action`
 + `is_valid_observation`
- The agent contains:
 + any observation and input augmentation methods
 + batching
- During training:
 + Dataloader creation:
  . Initialize the dataloader call
  . Create a proxy Dataset that wraps around the Environment's dataset, combined with
    the input and label processing defined by the agent
 + For each iteration: (1) dataloader gets observations and actions, (2) the agent learns using `.learn`
- During inference:
 + For each iteration: (1) dataset prepare suitable indices, (2) create dataloader, (3) dataloader gets observation and actions, (4) dataset compute correct/incorrect, (5) the mentor logs
- A good module should be able to work with:
 + Meta-learning
 + Model distillation, quantization
 + Evolution
 + Reinforcement learning problem
- Can use model card for declarative definition