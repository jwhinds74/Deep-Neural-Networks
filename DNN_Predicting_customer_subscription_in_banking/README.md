# __PREDICTING CUSTOMER SUBSCRIPTION IN BANKING: A NEURAL NETWORK APPROACHES__

![Image of Banking Transaction 1](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/9800577a-1011-490e-81a0-7d0b4489f540)

## __Below, you will find a walkthrough of the main modules from the notebook, highlighting the most relevant steps in this data science project. These steps enabled us to model the probability of customers contracting new financial services using a logistic regression model.__

#  __MAIN GOAL__
__The proposed project of modeling under Deep Learning approaches has been aimed at leveraging informed decision-making in the marketing areas of financial institutions. Therefore, the main objective seeks to predict the probability that institution's customers will contract new financial products and services during campaign events, thereby understanding how effective these campaigns have been..__

## __Dataset Description__
About Dataset the Bank Client Attributes and Marketing Outcomes dataset taken from Kaggle, offers a comprehensive insight into the attributes of bank clients and the outcomes of marketing campaigns. It includes details such as client demographics, employment status, financial history, and contact methods. Additionally, the dataset encompasses the results of marketing campaigns, including the duration, success rates, and previous interactions with clients. This dataset serves as a valuable resource for analyzing customer behavior, optimizing marketing strategies, and enhancing client engagement in the banking sector.

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/8a251804-f936-4949-92ce-8bdd2a1ae84e)

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/3765b7b5-ed25-4839-a106-e54aaaa60398)


## __Scheme of the general approach adopted for the development of the analysis__
To articulate a classification problem using the field “y” as a predictor, we need to identify the target variable or the variable we want to predict. In this dataset, it seems that the field y represents whether a client subscribed to a product or service offered by the bank (e.g., term deposit). Typically, in banking marketing datasets like this, y often represents the outcome of a marketing campaign, such as whether a client subscribed to a new banking service.
So, the classification problem here would be predicting whether a client will subscribe to the product or service (y = "yes") or not (y = "no").

Given this classification problem, we can use various machine learning approaches, but some common ones include the Logistic Regression: It's a simple and effective algorithm for binary classification problems like this.

## __Exploratory Data Analysis__

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/52d422b6-8ab5-470f-9ea8-c0b8b06e7071)  

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/9dc25f5a-24b6-4674-99f5-bee4c953430f)

## __Summary of Data Exploration, Cleaning and Feature Engineering__
As I went in deep into my modeling process, each step was meticulously executed to ensure the robustness and reliability of this analyses. Here's a comprehensive overview of my approach for handling data that is utilized across several variant for  Logistic Regression approaches:

## __Data Exploration:__
Examined the dataset to understand its structure and characteristics.

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/f73ae862-b30d-40de-a2d2-28582ed7e022)

### __Actions Taken for Data Cleaning:__
Handled missing values by dropping rows with missing values.
![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/8bd1d957-5ba2-42d6-a716-05f5a969bfcc)

### __Outliers Detection:__

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/40bd4f95-9d27-4cf8-bb9c-d7f6faa0903f)

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/8a7add41-6db7-4e43-9b7e-c05a7ac84996)

Employed Winsorization and robust treatment methods to mitigate outliers.

### __Feature Engineering:__
Defined numeric and categorical columns.
Created preprocessor pipelines for data preprocessing.

### __Dataset Split__
Let's split the dataset into a training and a testing dataset. Training dataset will be used to train and (maybe) tune models, and testing dataset will be used to evaluate the models. Note that I may also split the training dataset into train and validation sets where the validation dataset is only used to tune the model and to set the model parameters.

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/958f0522-0bed-4b1f-aa16-ccb79c8e54fc)

### __Feature Engineering__
Next, let's process the raw dataset and construct input data X and label/output y for logistic regression model training.

![image](https://github.com/jwhinds74/supervised_machine_learning/assets/15615882/222ac2b5-8e9e-44b9-af98-0bf54672d313)
