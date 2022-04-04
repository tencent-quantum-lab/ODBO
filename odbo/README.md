## odbo source code descriptions 
This folder contains the source codes for ODBO algorithms
* [bo.py](bo.py): Naive Bayesian optimization to generate the next set of queries points
* [featurization.py](featurization.py): Generate the feature vectors for different scenarios of protein datasets 
* [gp.py](gp.py): Gauassian process regression model constructions, inclding the GP with GP likelihood and GP with studentT likelihood.
* [initialization.py](initialization.py): Algorithm to find suitable initial set of measurements to be measured in experiments
* [plot.py](plot.py): Plotting functions to plot the confusion matrix for XGBOD accuracy and BO curves
* [prescreening.py](prescreening.py): The XGBOD search space prescreening algorithm
* [regressions.py](regressions.py): Surrogate modeling with GP or RobustGP
* [run_exp.py](run_exp.py): Wrapped functions to pack the BO search of each iteration
* [test.py](test.py): Test functions to make sure the package is installed correctly
* [turbo.py](turbo.py): Trust region Bayesian optimization algorithm to generate the next set of queries points
* [utils.py](utils.py): Useful functions 
