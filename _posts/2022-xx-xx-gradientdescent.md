---
title: "Parameter Estimation with Gradient Descent"
published: false
tags: parameter-estimation model-fitting
description: "Basic Entry to Parameter Estimation"
---
by: [Chuan Fu Yap](https://chuanfuyap.github.io)

# Parameter Estimation aka _Model Fitting_ aka _Model Learning_
 
For those unfamiliar, model fitting means estimating the regression coefficients, also known as the model parameters. Other phrases that could be used interchangeably with model fitting are parameter estimation and model learning.

> model fitting = model learning = parameter estimation

In some field they would only use one of these terms but they are basically the same idea, and they would be phrased as such when training a model,
- fitting model to data
- model learning from data (some would then refer model as learner)
- estimating the parameters of the model.

#### Loss function/cost function/objective function
Practice of model training falls under an optimization problem. Where optimization here means we are trying to obtain the best possible solution for a function. Reason I say best possible solution, is that typically in an optimization problem we cannot obtain a closed form solution easily. That is to say we cannot move the variables around easily and get the perfect answer. Instead we optimize towards it, trying to minimise or maximise the function. The either min or max choice depends on the function we use. As before, there are various functions that are used and different field uses a different one but the idea is all the same. The function is used to score the model being trained. And for example a loss/cost function computes the error between actual outcome vs model's outcome, therefore we want to minimise the function as we would want to reduce the error in the model. Alternatively, we have an objective function to determine how well the model is doing, we would then want to maximise the function. 