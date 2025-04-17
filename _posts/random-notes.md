Old concept of understanding interaction terms. New and better one is now in the blogpost. 


Previously, model interpretations have been on unit increase, how would that apply here? When we introduce the interaction term, basic unit increase of coefficient per variable no longer applies for individual variables, and we can only interpret the interaction term. To make sense of it numerically, we would have to make some pseudo data and use them with the model and do more maths. 

First we need to generate the following values for the two variables:

|   x |  fruit  |
|:----:|:--------:|
|   0 |       0 |
|   1 |       0 |
|   0 |       1 |
|   1 |       1 |

> x here is continous variable, but we set it as 1 as a basic unit increase
>
> fruit is categorical variable, where 0 is used for apple, 1 for orange

Next, we can them in to model and make four different predictions, which would give us the estimated unit increase when these two variables interact. For example in the first row when both values are zeroes, we can get estimated $$Y$$ value when the $$X$$ variable is not present, but _apple_ is present . In the second row, we would have the unit increase when $$X$$ and _apple_ present and so on. 

Following that, to numerically observe "interaction" or "dependency", we would have to narrow down on group of estimated values. For example, for $$X$$ with 0, we take values from orange/apple values, and take the difference between the two values. This would give us the the is a unit difference between the fruit variables (orange vs apple) when $$X$$ is 0. We can again isolate when values of $$X$$ is 1, and repeat the difference between the two variables. Given that $$X$$ is a continous variable, i.e. not limited to just 0/1, we can repeat this process for other numerical values, and we can see that as $$X$$ changes, the unit difference between orange and apple are different, i.e. a fruit variable depends on $$X$$. Whereas, if we did this exercise of check each possible values of fruit variable for different $$X$$ in an additive model, the unit difference would remain the same, which would be the coefficient of the fruit variable. To make sense of this, we can refer back to the plots above, and move along the x-axis and observe that difference between orange and apple's _y_ value.
