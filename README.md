# Household power consumption estimation

The [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) dataset from the UCI Machine Learning repository was used to train and test a set of regression models. The set of algorithms includes Linear Regression, L2 Regularization (Ridge), Elastic Net, Decision Tree, and Random Forest. The last one demonstrated the best performance. As a result, Random Forest is used in [the user-friendly demonstration](https://power-consumption-prediction.streamlit.app/).

#### Data preprocessing
Several data preprocessing practices were applied before designing and developing the models. The steps include:
- data cleaning and handling missing values
- feature selection and extraction
- standardization

#### Data split
70% of the dataset was used for the training set while 30% for the test set.

#### Models' evalutation

###### - Linear Regression and Ridge evaluation on the test set

```
    Mean Squared Error (MSE) = 0.0016276345367416988
    Mean Absolute Error (MAE) = 0.025779443969332457
    Coefficient of determination = 0.9985476216083097
```

##### - Elastic Net evaluation on the test set

```
    Mean Squared Error (MSE) = 0.47365937658049345
    Mean Absolute Error (MAE) = 0.5360517948097412
    Coefficient of determination = 0.5773420703248616
```
##### - Decision Tree evaluation on the test set

```
    Mean Squared Error (MSE) = 0.0012052484970331043
    Mean Absolute Error (MAE) = 0.014439103815323762
    Coefficient of determination = 0.9989245270764454
```

##### - Random Forest evaluation on the test set

```
    Mean Squared Error (MSE) = 0.0010795055934000426
    Mean Absolute Error (MAE) = 0.01783327091618699
    Coefficient of determination = 0.999036730566862
```

N.B: The GitHub-actions bot is used to push an empty commit every 10 hours in order to keep the demo app awake. 
