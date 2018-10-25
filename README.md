# Purpose

**Make it easy to create, to trace, to debug, and to recreate models/algorithms.**

What does that means:
- Incorporation of common components (res-block, seblock,...) that can easily be used and constructed (which is good for model creations and modifications).
- Data processing in functional form, which allows clear knowledge of what happens to the data before it is fed to the algorithms/models.
- A tool to view and debug:
    + view model statistics
    + tinkering with the data and see what happens at the result (top-k result)
- Inference must be constructed inside the model, with the input is the most basic data point (thinking of a complete stranger who needs to use your model, that person will not know anything about the nit-picks of your models and your data, they only have a data point and want to see the result coming out of your model)

# Requirements

Dawnet requires `pytorch` and `opencv` to work properly. Since many distributions exist for these libraries, we recommend user to install themselves to avoid messing up the environment. If you don't have `pytorch` or `opencv` installed, then you go to https://pytorch.org to install appropriate version, and `conda install -c conda-forge opencv` to install `opencv`.


# Roadmap to usability

- Session must work
- Ability to get batch of data
- Summarize session and model information
- Implement mixup
- Test all convs architecture


# Model

- A model should have the evaluate method ready (this part should be abstracted away from the progress / training procedure)
- A model should have the load method ready (only for inference, because continual training requires knowledge about optimizer, training iteration)


# Data

Model is not the only part in creating intelligent system. Data plays a vital role in this process too. A lot of time, playing around with data, seeing how the model behaves when data is tweak a little bit can provide crucial insights for model improvement. Hence, data manipulation must be made easy to use.