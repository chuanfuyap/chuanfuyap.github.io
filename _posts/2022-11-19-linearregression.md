---
title: "Linear Regression Crash Course"
published: false
tags: linear-model linear-regression
description: "Theory dive with application in Python - 28 min read"
---

tldr; This will be a crash course on Linear Regression for statistical analysis with code implementation in Python. It will have minimal explanation on the maths. Feel free to navigate to [Table of Contents](#toc) to skip to section of interest which would have theory and code on them, or [click here](https://github.com/chuanfuyap/mini-ds-projects/blob/main/linear-regression/regression-notes.ipynb) for code compilation. 

# Prologue
Linear regression is what I would describe as the 'swiss army knife' of statistical models or the ['Canon in D' of modelling](https://www.youtube.com/watch?v=JdxkVQy7QLM) if you will. This simple equation:
\begin{equation}
y = mx+c
\end{equation}
is often undersold when you were first taught it in high school level (or earlier I don't know what kind of education you had). We were told _y_ is what we want to map to using a line of best fit with point _x_, with which we use to compute slope/gradient/rate of change _m_ along with the intercept _c_. As we move up in education levels, its significance is elevated and the equation is rewrote as the following:
\begin{equation}
Y_i = \beta_0 + \beta_1 X_i + \epsilon_i
\end{equation}
This time, it is introduced as a statistical tool with the usual package of _p-value_ in it along with an error term. This form of equation is also known as simple linear regression (SLR), for it has single independent variable _X_ relating to single dependent variable _Y_. 

While this equation have been around for hundreds of years, its application is becoming more widespread across scientific research as well as analytical departments of businesses in recent century. The reason for this is the (extreme) generalisation this equation allows for (along with its interpretability), which is normally not taught to us, at least not in an adequate manner. 
>By generalisation, I mean how data can be generalised in this linear manner when not all data relationship is linear, as well as the ease of modification to the model to expand on its usage. 

We are typically just told the dry version of linear regression usage which is more or less:
> A deterministic model describing the **linear relationship** between two variables _X_ and _Y_. 

Usually, _X_ is known as the independent variable and _Y_ is the dependent variable, because its value _depends_ on another variable(s) (e.g. _X_) or parameter(s). There are alternative semantics used by different fields such as in machine learning, _X_ is feature and _Y_ is target, more can be read on this [wiki page](https://en.wikipedia.org/wiki/Dependent_and_independent_variables). 

> dependent variable = response = outcome
>
> independent variable = predictor = features

This blogpost will cover the basic of linear regression and some of its extensions that makes it the swiss army knife of statistical models. Another name for this modelling approach which I prefer is **linear model** because _regression_ is an approach in machine learning which predicts continuous outcome, linear models can be generalised beyond that as some of you would be familiar with logistic regression which predicts binary outcomes. 

This post is the cumulation of knowledge I have gathered from [STAT501](https://online.stat.psu.edu/stat501/), a PennState course module, [Statistical Rethinking](https://www.amazon.co.uk/Statistical-Rethinking-Bayesian-Examples-Chapman/dp/1482253445), a wonderful statistics textbook and finally [Coursera's Machine Learning course (pre 2022 update)](https://www.coursera.org/learn/machine-learning) by Andrew Ng. So this blogpost also serves as my notepad for collating the notes I have taken from these sources as well as quick application of the theory in Python. 

<a class="anchor" id="toc"></a>

### Table of Contents
- [Simple Linear Regression](#theory)
    - [What is SLR?](#what)
        - Assumptions
        - Interpretation
    - [Why use SLR?](#why)
        - Hypothesis Testing
        - Prediction
    - [How to know SLR is working?](#how1)
        - Coefficient of Determination, R2
        - Check assumptions are not violated
    - [How to estimate parameters in SLR?](#how2)
        - Linear Algebra solution
        - Gradient Descent
- [Multiple Linear Regression](#mlr)
    - Confounders
    - Interaction Models
    - Model Building Process
- [Transformations on _X_](#transformation)
    - Log
    - Polynomials
- [Beyond Linear Regression](#beyond)
    - Logistic Regression
    - and more
- [Summary](#summary)
- [Appendix](#appendix) (if I used a word without explaining it, it is probably here)
    - Statistical Semantics/Glossary
    - Formulas

<a class="anchor" id="theory"></a>
<a class="anchor" id="what"></a>

## What is Simple Linear Regression (SLR)?
\begin{equation}
Y = \beta_0 + \beta_1 X + \epsilon
\end{equation}

- _Y_, is the response/dependent variable
- _X_, is the predictor/independent variable
- $$\beta_0$$, is the intercept
- $$\beta_1$$, is the regression coefficient/slope/gradient
- ϵ, is the normally distributed random error

This is a 'simple' linear model because it has one predictor variable and one response variable. For multiple linear regression (MLR) with > 1 predictor variable, go [here](#mlr). 

#### Assumptions/Conditions for SLR to work
- **L**inear relationship/function
- **I**ndependent errors
- **N**ormally distributed errors
- **E**qual variances

It is an acronym (LINE) to help you remember, since the model gives you _line_ of best fit. 
![lm-info](https://raw.githubusercontent.com/chuanfuyap/mini-ds-projects/main/linear-regression/lm-info.png?raw=true){:class="img-responsive" height="65%" width="65%"}{: style="display:block; margin-left: auto; margin-right: auto;"}

Plot above is a brief anatomy of a SLR model highlighting the components from the equation. Note, I have plotted the data points as well as the line of best fit resulting from the model. Whenever possible visualisation of the model along with the data points is important when the analysis is being presented. For example in the plot above we can see there is a potential outlier. In another example plot below, while you can get a straight line to fit, it might not be best representation of the data which shows a curvature. 

![lm-info2](https://raw.githubusercontent.com/chuanfuyap/mini-ds-projects/main/linear-regression/ht_wt2.png?raw=true){:class="img-responsive" height="65%" width="65%"}{: style="display:block; margin-left: auto; margin-right: auto;"}

Above plots can be made easily using [seaborn](https://seaborn.pydata.org) with the following code `sns.lmplot(x, y, data)`

<a class="anchor" id="coef"></a>

### Interpretation of SLR
##### Slope/Gradient/Regression Coefficient
> β1=1, 1 unit increase in _X_ results in 1 unit increase in _Y_

- We expect the mean response _Y_ to increase/decrease by estimated $$\beta_1$$ unit for every **one** unit increase in _X_. Substitute _X_ with 1 in the equation and it'll make sense in the equation. 
- In some cases we standardise the data _X_ by subtracting the data vector _X_ by its mean and dividing by its standard deviation; in code `(X-X.mean())/X.std()` assuming _X_ is an array. The coefficient interpretation changes to mean change in standard deviation in the association between _X_ and _Y_. 


##### Intercept
> mean value of _Y_

- The mean value of _Y_ when _X_ is zero. Substituting _X_ with 0 and it would make sense in the equation. Usually data you are analysing the value of _X_ theoretically cannot be zero, for example in height and weight relationship, neither variables can be zero. So to get an accurate interpretation for situations like this, we can mean-centre the _X_ prior to model fitting, and the intercept would give the mean _Y_ of the data when _X_ is at its mean. Mean-centre means subtracting data vector _X_ with the mean of _X_; in code `X - X.mean()`, assuming _X_ is an array. 


<a class="anchor" id="why"></a>

## Why use SLR??
The simple interpretation of unit increase from coefficient explained above is one the reasons on why we use SLR, but the other reasons are:

### Hypothesis testing
> null hypothesis, β1=0
> 
> alternative hypothesis, β1=/=0

Hypothesis testing in frequentist statistics require a null hypothesis, which is something we want to disprove. With SLR, the null hypothesis is `β1=0`, therefore the thing we want to show is that `β1=/=0`, which is the alternative hypothesis. This would demonstrate that there is an association between _X_ and _Y_. This also means that a change in _X_ results in a meaningful (or statistically significant) change in _Y_. 


We can do this using [statsmodels](https://www.statsmodels.org/stable/index.html), for example:

```python
import pandas as pd
import statsmodels.api as sm
# load data
df = pd.read_csv("data/student_height_weight.txt", sep='\t')
## setting the variable names
## For X we need to run this function to generate a columns of 1 to represent the intercept
## you can check using X.head()
X = sm.add_constant(df.ht) 
y = df.wt

## build OLS object, which stands for ordinary least square
## a type of regression model fitting method. 
model = sm.OLS(y, X)
## fit the model
results = model.fit()
```

To view the outcome of fit execute `print(results.summary())` and we get the following table with lots of information. 
![lm-output](https://raw.githubusercontent.com/chuanfuyap/mini-ds-projects/main/linear-regression/lm-output2.png?raw=true){:class="img-responsive" height="80%" width="80%"}{: style="display:block; margin-left: auto; margin-right: auto;"}

I have circled the immediate things that would be of interest for simple models, everything else is more advanced for further model diagnosis if necessary. 

From top to bottom and left to right, things of interest are:
- $$R^2$$ **(R-squared), Coefficient of Determination**
    - explains variation in _Y_ explained by the model, values ranges from 0-1, and higher value is better. But a _good_ value depends between research field. 
- **Coefficient (coef)**
    - this was as explained [above](#coef), where const correspnods with the intercept's coefficient, and _ht_ is _X_ of this model.
- **P-value (P >\|t\|), the all fabled measure of statistical significance**
    - the essential ingredient of frequentist statistics where values <0.05 implies statistical significance where **null hypothesis** is disproven and (usually) therefore an important finding.
- **95% Confidence Interval (0.025, 0.975)**
    - The measure of confidence on the estimated _coef_. That is, we are 95% sure the estimated _coef_ is between these values. 

Normally, when we report our results, on top of plotting, we would **highlight the three things within the blue box**, which are _coefficient value_, _p-value_ and the _confidence interval_. 

CAUTION:
- large $$R^2$$ does not necessarily mean it is a good model.
- low/significant _p-value_ is not useful if the coefficient is low. 

### Prediction
Arrival of big data allowed for the boom of machine learning, and linear model is one of the algorithms used in supervised learning for prediction purposes. Models trained with `statsmodels` can be used for prediction as well using `predict` function on the fitted model object, e.g. `results.predict(new_X)`. 

The more popular and convenient option for prediction is with [`scikit-learn`](https://scikit-learn.org/stable/). Example code:

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

## load data
hw = pd.read_csv("data/Howell1.csv", sep=';')

## set variables names
X = hw[['height']]
y = hw[['weight']]
## fit and train model
lm = LinearRegression()
lm.fit(X,y)
```

The above code would give you a trained linear model for prediction when you have new data, e.g. `lm.predict(new_X)`, you would then be able to score it using any metric you need, and `scikit-learn` provides many of it on their [website](https://scikit-learn.org/stable/modules/model_evaluation.html). 

> Disclaimer, there is more to good machine learning practice than just fitting a model and scoring it. 

<a class="anchor" id="how1"></a>

## How to know SLR is working?
We have learned what is SLR and what it's used for, now we learn how we know it is working as intended. 

### Coefficient of Determination, $$R^2$$ 
This gives the measure of how much variance the response variable is explained by the model. Higher value would usually imply a good model, however this should always be accompanied with a visual inspection with a scatter and linear plot as a single data-point can impact the $$R^2$$. 
### Making sure assumptions are not violated 
The **LINE** assumption is of extreme importance when it comes to hypothesis testing. The statistical significance would not hold if any of the assumption is violated, because these tests were developed with these assumptions in mind. When it comes to prediction, overfitting and underfitting is priority over model assumptions. 

There are various tests for this, coming in quantitative and qualitative flavors. I personally prefer the visualisation approach for this, with these three that covers all the assumptions:

1) Residuals vs Fits plot
![resid-fit](https://raw.githubusercontent.com/chuanfuyap/mini-ds-projects/main/linear-regression/resid_fit.png?raw=true){:class="img-responsive" height="90%" width="90%"}{: style="display:block; margin-left: auto; margin-right: auto;"}

This is a plot where the x-axis is the fitted values, values predicted by the model on data and y-axis is the residual of the model, where residual is the difference between actual _Y_ and predicted _Y_. This lets us check for **_linear_** and **_equal variance_** assumption.

To plot this:
```python
## load data and fit model
alcohol = pd.read_csv("data/alcoholarm.txt", sep="\s+")
X = sm.add_constant(alcohol.alcohol)
y = alcohol.strength

model = sm.OLS(y, X).fit()

## extract values to plot
fitted = model.predict(X) #get predicted/fitted values
resid = model.resid ## gets us the residual values

## scatter plot
sns.scatterplot(fitted, resid) 
# np is numpy package used to find min max value
plt.hlines(y=0, xmin = np.min(fitted), xmax=np.max(fitted), linestyle="--") 
```

When the residual bounces randomly around the horizontal (residual=0) line such as plot above, it would mean the **_linear_** assumption is reasonable. If it was violated, it would have an obvious pattern such as a fixed line or wave for example. 

When the residual forms a approximate 'horizontal band' along the horizontal line, it would mean the **_equal variance_** assumption holds. If it was violated, it would form a cone in either direction. 

2) Residual vs Order plot
![resid-order](https://raw.githubusercontent.com/chuanfuyap/mini-ds-projects/main/linear-regression/resid_order.png?raw=true){:class="img-responsive" height="90%" width="90%"}{: style="display:block; margin-left: auto; margin-right: auto;"}

This is a plot where the x-axis is the order of the variable was collected, and y-axis is the residual of the model, where residual is the difference between actual _Y_ and predicted _Y_. This lets us check for **_independence_** assumption. This is mostly use when you have a time series data where the order of the data is known, and we want to know if the response variable is dependent on the time variable. This plot is really only appropriate when we know order the data is collected such as the time. 

To plot this:
```python
## load data and fit model
eq = pd.read_csv("data/earthquakes.txt", sep="\t", index_col=0)

X = sm.add_constant(eq.index)
y = eq.Quakes

model = sm.OLS(y, X).fit()

## extract values to plot
resid = model.resid

## scatter plot
sns.scatterplot(eq.index, resid)
# np is numpy package used to find min max value
plt.hlines(y=0, xmin = np.min(eq.index), xmax=np.max(eq.index), linestyle="--")
```
When there is a random pattern such as the residual vs fit plot, it would mean no serial correlation or dependence. The example plot shown reveals a wave pattern in the data points, which means there is a trend of the response variable dependent on time. 

When this happens, SLR is no longer the ideal analysis tool for this dataset, we should instead move on to time series analysis using for example autoregressive models. 

3) Normal probability plot of residuals (QQ-plot)
![qqplot](https://raw.githubusercontent.com/chuanfuyap/mini-ds-projects/main/linear-regression/qqplot.png?raw=true){:class="img-responsive" height="90%" width="90%"}{: style="display:block; margin-left: auto; margin-right: auto;"}

This plot shows the residual's quantile against a theoretical 'normal' quantiles. This lets us check the **_normality_** assumption. If the data or residuals are normal, it would match up closely with the theoretical line. Deviations of data points from the red line would indicate non-normal residuals/data. 

This can be plotted as such:
```python
fig = sm.qqplot(resid, line="s") ## built into statsmodel
```

Above are some of the easy checks for linear model assumptions. `statsmodels` have more built in functions for other plots and quantitative tests, for example this [page](https://www.statsmodels.org/dev/examples/notebooks/generated/regression_plots.html) covers other plots that can be used to diagnose/investigate the model, as well as a simple one line code that plots several things, `sm.graphics.plot_regress_exog(fitted_model, "variable_of_interest")`

<a class="anchor" id="how2"></a>

## How to estimate the parameters in SLR?
This section is probably not of interest to many since I have already shown how `statsmodels` and `scikit-learn` can fit the model with one line. But if you are interested in how this is done, below are two methods that are used for linear model fitting. 

> REMINDER: regression coefficient = slope = gradient, is the parameters in SLR model. 

In SLR, when estimating parameters the most common method is to find parameters that would produce the minimum sum of squared residuals. This method is known as _ordinary least squares (OLS)_.
![qqplot](https://raw.githubusercontent.com/chuanfuyap/mini-ds-projects/main/linear-regression/resid_vis2.png?raw=true){:class="img-responsive" height="65%" width="65%"}{: style="display:block; margin-left: auto; margin-right: auto;"}

Figure above visualises the residuals in a SLR model, which is the gap between predicted value (line) and actual value (dot). 

### Linear Least Square/Normal Equation
With OLS, there is a [closed form solution](https://en.wikipedia.org/wiki/Closed-form_expression) for this:

\begin{equation}
\hat{\beta} = (X^TX)^{-1}X^Ty
\end{equation}

Where 
- $$\beta$$, the model coefficients being estimated
- $$X$$, the independent variable matrix
- $$X^T$$, transposed independent variable 
- $$y$$, the dependent variable
- the superscript of -1, is notation of inverse matrix

### Gradient Descent
This method can be quite involved and deserves a post of its own, so I would just mention that _gradient descent_ is or a variation of it, the _stochastic gradient descent_ are the commonly used algorithms to estimate parameters in SLR. 

[placeholder for future blogpost]() 

<a class="anchor" id="mlr"></a>

## Multiple Linear Regression
Now we move beyond the 'simple' in SLR and include more than one independent variable in the model, with that we get Multiple Linear Regression (MLR). 

More information is 'always' helpful when it comes to modelling, which is why MLR is preferred if there is extra variables available. 

> `always` have been placed in quotes because there can be cases that more information just leads to an overfit and misinformed model. Model building should always be done with some level of caution instead of putting all the data in and expect the best. An age old wisdom in model building is '_garbage in, garbage out_'. 

Below are the statistical explanations on why using MLR is useful. 

### Control/Adjust/Account for Confounders
The most commonly used if not most useful reason to have additional predictor variables in linear models is to allow the model to _control/adjust/account_ for confounders. I use the three phrases: _control_, _adjust_ and _account_ because depending on what field of research you are in, different semantics are used.

> What is a _confounder_ and why should we control for it?
>
> A confounder is a variable that has an effect on both dependent and independent variable. This allows the confounders to hide important effects and potentially produce false ones. 

Therefore, by including the confounder in the model, we have controlled for it, this would "separate" their effects from the other independent variables. 

Further to this line of thought, it could also be that other variable(s) are known to associated with dependent variable (but not to independent variable), we could also include it in the model, this is referred as '_conditional analysis_', where we condition the model on the known variable to analyse other variables. This would have the same effect as before, where we can then see effect of the new variable being studied after the known variable's effect is accounted for. 

Given the extra variables in the model, the interpretation is updated slightly as such: 

> each $$\beta$$ unit results in a unit change in mean response _Y_ provided all other predictors are held constant. 

To work with an example that would be easy to understand, let's continue with the theme of height. The fantasy race [dwarves](https://en.wikipedia.org/wiki/Dwarf_(folklore)) are known to be shorter than humans, and visually pudgy, so I am assuming for them to have higher BMI than humans. If we were in Middle-earth collecting census, without accounting for race, our weight/height data would be rather strange. 

> PICTURE! INSERT DWARF PLOT SIDE BY SIDE

Plot on the left shows what happens to our model when we fail to account for race, and used only one variable. The plot on the right shows two clear parallel line between them when accounting for race.

i.e. the model on the right is:

\begin{equation}
weight = \beta_0 + \beta_1 height+ \beta_2 race
\end{equation}

This toy model can be "_thought of_" as having 'two' models in one. That is one regression model for each race. Race here is a categorical model which would be represented with 0 or 1 for either races, e.g. 0 for _dwarf_, 1 for _human_. When we want to compute height of _human_, $$\beta_2$$'s value would be added to the calculation of weight, and when computing for _dwarf_'s weight, $$\beta_2$$'s value is not included.

E.g.

For _human_ (coded as 1):

\begin{equation}
weight=\beta_0 + \beta_1 height+ \beta_2 1
\end{equation}

For _dwarf_ (coded as 0):

\begin{equation}
weight=\beta_0 + \beta_1 height+ \beta_2 0
\end{equation}

> 0 multiply $$\beta_2$$ is still 0. so the outcome of weight is just $$\beta_0$$ + $$\beta_1$$*height

So this model would compute the overall relationship between height and weight (in $$\beta_1$$), and include the effect of race within it (in $$\beta_2$$). If the labels of race is flipped, 1 for _dwarf_ and 0 for _human_, $$\beta_2$$ would just have a negative value, but results is still the same. 

Above is an easy to understand/visualise example with categorical predictors (_dwarf_ vs _human_) where we can see an _additive effect_ of the extra '_race_' variable. 

> When using **categorical predictors** in a linear model, stats packages would assign one of the values as a reference/template/contrast (once again it is field specific for the semantics). It would code the reference value 0 and 1 for the other during modelling. For example, in the example above, _dwarf_ is the reference variable, and the _human_ variable would have an unit increase in height given the same weight value between the two races. 

To make it clear, use of extra variables in linear models is not limited to categorical predictors as per example above, quantitative variables (continuous or discrete) can be used in the same manner. Further, you can include >2 variables in the model despite the example, e.g. for height weight relationship, you can include age, and sex. 

> **CAUTION**: the number of predictors (p) we can have in a model is limited by the number of samples (n). That is, we cannot have a `p>n` situation in our models. For statistical reason because we would not have enough degrees of freedom to estimate the parameters in the model. In machine learning, for linear models, this would likely lead to overfit and it learns the noise and reduce model generalization in prediction.

### Interaction Models
When the importance/effect of one variable depend upon another variable, this is known as _interaction effect_. Models that include this would then be interaction models, and as before, the model can include >1 interaction effect within it. 




### Model Building Process
In SLR, I highlighted $$R^2$$ as the metric to know if the model is "correct" or useful. This metric is also applicable for SLR, however, as mentioned above we can't merely rely on a high $$R^2$$ alone in deciding a good model, and to visualise it. When it comes to MLR, visualising it can be challenging as the number of dimensions have increased. Therefore we would now need to rely on other **metrics** to inform us if our MLR model is "correct"; these are AIC, BIC and log-likelihood, all these information are provided in the output from statsmodel shown above when you execute `print(results.summary())`, located just below `R-squared` value. 

#### Model Comparison
Aside from using the above metrics alone, another way to make sure model is useful or 'correct' after adding a variable or even an interaction effect is to perform model comparison that can be passed through a statistical test to generate a everyone's favourite p-value. There's two options for this:
* General Linear F test/ANOVA
* likelihood-ratio test (LRT)

But the core idea is the same, which is, we build two models:

i) a 'full model' including all variables/interaction of interest, 

ii) 'reduced model' or 'restricted model' with the extra variable/interaction removed. 

Then we compare it statistically, and decide which model to keep. This statistic is done with _F-distribution_ for F-test or _chi-squared distribution_ for LRT. The null hypothesis here is that the reduced model is _better_, therefore a 'significant' p-value indicate the full model (extra variable/interaction) is _better_. Better can be rephrased as _explain the data/Y better_. 

The following example code is how you would perform a _LRT_:
```python
### extract log-likelihood of fitted models
full_llf = fullmodel.llf # where fullmodel is statsmodel object with full model fitted
reduced_llf = reducedmodel.llf # as above but for reduced model, e.g. one variable less

### ratio, but since it is log-transformed we can take the different
likelihood_ratio = 2 * (full_llf - reduced_llf)

### take degree of freedom (dof) from models to be used in the stats
full_dof = full_llf.df_model
reduced_dof = reduced_llf.df_model

dof = full_dof - reduced_dof

### compute the p-value
from scipy import stats # we use scipy for the statistics function
p = stats.chi2.sf(likelihood_ratio, dof)
```



<a class="anchor" id="transformation"></a>

## Transformation on _X_

### Log
mention change in interpretation after log-transforming a variable [source](https://data.library.virginia.edu/interpreting-log-transformations-in-a-linear-model/)
### Polynomials



<a class="anchor" id="beyond"></a>

## Beyond Linear Regression
At the start of this post, I called linear model the 'swiss-army knife' of models, not just because of all its applications and versatility granted from data transformations above, but also for its generalization on the type of data it can model. For those that took basic stats course would know the examples I have been giving are modelling only continuous data, but there are other kinds of data such as discrete, categorical and ordinal. In my MLR example, I talked about categorical data as an independent variable, but this can also be modeled as a dependent variable. To achieve this with linear model, we rely on link function to transform the outcome into a range between 0-1 and we can further dichotomise it into a categorical outcome by choosing a cutoff e.g. 0.5 to split into 0/1. When dealing with binary outcome, this is a simple _logistic regression_, and the equation is the following:

\begin{equation}
Y\frac{1}{1+e^{-(\beta_0 + \beta_1 X)}}
\end{equation}

The core concepts learned above are mostly transferable, most importantly the interpretation have changed. Those familiar with probability would know we are now dealing with log-odds, the  $$\beta$$ is the estimated increase in the log odds of the outcome per unit increase in the value of the _X_. To make it easier to understand, we can use transform $$\beta$$ with the exponential function $$e^{\beta}$$ in order to obtain the odds ratio associated with a one-unit increase in the _X_.

> What is _odds ratio_ (OR)?
>
> Every unit increase is 'OR' many times more likely to to have the outcome. E.g. :
> * if OR is 4, every unit increase is 4 times more likely to have the associated outcome. 
> * if OR is < 1, means the outcome is less likely to occur. 
> * if OR=1, there is no increase or decrease in likelihood of outcome, i.e. no association between _X_ and _Y_. 

Sometimes $$R^2$$ is not computed, but usually a pseudo $$R^2$$ is provided which provides a similar interpretation, this is also visible in the `model.summary()` function. 

To build logistic model, I recommend to use the `GLM` function in `statsmodels` over the `Logit` function. Example code:

```python
## usual formatting of y/X data for statsmodels
X = sm.add_constant(df.X) 
y = df.y

## creating and fitting model
model = sm.GLM(y, X, family=sm.families.Binomial())
results = model.fit()
```

#### Generalized Linear Model
So what is this `GLM` that I prefer over `Logit`? GLM stands for generalized linear models, which is a this modelling framework that extends regression models to include many more data types than just continuous, and the `Binomial` version I chose is used for categorical outcomes (_Y_). For other 'families' of outcome that can be modelled, please visit this [link](https://www.statsmodels.org/stable/glm.html) which is extension through link functions.

#### Linear Mixed Effects Model
This is a variation of modelling that allows us to model dependent data (which violates one of the assumptions) that are generated from longitudinal data, i.e. samples that are collected over time multiple times. For example studying someone's exercise and diet, you would have multiple observations on their weight, calories burned and food intake. This is where [linear mixed effects model](https://www.statsmodels.org/stable/mixed_linear.html) come in. Please follow the link to learn more on this as this is an advanced subject. Alternatively, follow this [blogpost](https://www.tjmahr.com/plotting-partial-pooling-in-mixed-effects-models/) for useful visualisation example, but in with example code written _R_. 

#### Time series model
Another advanced modelling method to deal with repeated observations over time is known as time-series analysis, and this is done with an [autoregressive models](https://www.statsmodels.org/stable/tsa.html#module-statsmodels.tsa). 

#### Survival analysis
Another thing we can learn from tracking things over time, is when would an event happen, classically, this would be _death_, which is why this field of study is known as _survival analysis_. Regression can be extended to perform [surival regression](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html) for use in survival analysis. 

There are more extensions in linear models that I have not listed because I am not familiar or even aware of their existence. But once you have a strong foundation within linear model alone, you can expand beyond it. 

<a class="anchor" id="summary"></a>

## Summary
Linear model is a class of statistical method that can be used to model many things in a 'general' manner. This general-ness can be expanded through data transformation such as polynomial transformations and interaction models as well as use of link functions. Its use is found in many applications with two popular python packages supporting two specialised use case, which are `statsmodels` for hypothesis testing and `scikit-learn` for machine learning/prediction modelling. I cannot stress enough on the 'generalness' linear model, as Richard McElreath puts it, this is a geocentric model for statistics, while it is not accurate, it is still very useful. Even if most data in the world is not 100% linear or Normally distributed, linear model still can be applied. This is why it is found in all fields of research. 

> Disclaimer: This whole post may feel very heavy in content, but it is still a crash course where some things are overlooked. So don't be surprised if there are still things that you do not understand when reading other literature on this subject matter. But I hope this gives you enough understanding to read a textbook and quickly fill in the blanks for yourself. 

<a class="anchor" id="appendix"></a>
 
# Appendix
## Statistical Semantics/Glossary
Useful statistics semantics in case you want to venture out and read more on your own

##### Population model/statistics/parameter
- The wonderful imaginary optimal true model with parameters that accounts for every single sample available within the population being studied, such as the whole world or a given country, etc, _which is often not known_. 

##### Estimated/Sample model/statistic/parameter
- The model built/estimated from limited (not all) samples of the population, which is what we really have, hence all the nuance calculations to include confidence interval and standard error etc. 

Population vs sample is an important distinction. For example, we are studying the height of dwarves, we would only be able to measure (sample) 500 of their height. From this 500 measurement we can determine the _sample mean_, where mean is the model parameter. That is we know the _mean_ of the samples measured but not _true_ mean of the dwarf population being studied. Which is why confidence intervals are often included in the reporting of parameters, to state that we are 95% confident sample mean is the (true) population mean. 

##### Degree of freedom (DOF)
- Not directly related to linear models, but an important factor in all statistical models. DOF is the number of 'information/observations' in the data that are free to vary when estimating statistical parameters. Usually `DOF = N - P`, where _N_ is the number of observations/samples available, and _P_ is the number of a parameters being estimated. This is an important information because if DOF is less than or equals to 0, no p-values can be computed, therefore _P_ cannot be more than _N_. 

##### Independent and Dependent Variable
In case you skipped prologue, there are field specific synonyms to independent/dependent, which is detailed in this [wiki page](https://en.wikipedia.org/wiki/Dependent_and_independent_variables).

##### Effect size
For Linear Models, the coefficients are the effect sizes of the phenomenon being studied. 

##### Close Form Solution
An oversimplification is that the equation can be solved analytically. That is we can move the variables around and we can compute the solution for a variable of interest easily.

E.g. in SLR we have simple `y = b*x + c`, if we know values of `y/x/b`. We want to solve for `c`, we just rearrange the equation to `c = y - b*x`, and put in the corresponding values to solve for `c`. 

##### Log-likelihood
A likelihood of a given model is a measure of how 'likely' or 'probable' this model is in generating the given data. I say 'given model' because this is a statistical term that can be applied to any statistical models. As this explanation suggests, this can be used to decide how useful the model is, and the higher the value the better (more likely) the model. Log transformation on the model's likelihood function would give you the log-likelihood. If you would like to know more on linear regression's log-likelihood function, please follow this [link](https://www.statlect.com/fundamentals-of-statistics/linear-regression-maximum-likelihood).

## Relevant Formulas
- mean of X/Y
\begin{equation}
\bar{x}/\bar{y}
\end{equation}
- predicted Y
\begin{equation}
\hat{y}
\end{equation}
- Intercept
\begin{equation}
\beta_0 = Y - \beta_1 X - \epsilon
\end{equation}
- Regression coefficient
\begin{equation}
\beta_1  =  \frac{\sum(x_i - \bar{x}) (y_i - \bar{y})}{(x_i - \bar{x})^2}
\end{equation}
- Residuals
\begin{equation}
resid = y_i - \hat{y}
\end{equation}
- Sum of squared residuals aka Sum of squared errors (SSE)
\begin{equation}
SSE = \sum(y_i - \hat{y})^2
\end{equation}
- Mean-squared error (MSE)
\begin{equation}
MSE  =  \frac{\sum(y_i - \hat{y})^2}{n-2}
\end{equation}
- Akaike information criterion (AIC)
\begin{equation}
AIC  =  
\end{equation}
- Bayesian informaiton criterion (BIC)
\begin{equation}
BIC  =  
\end{equation}