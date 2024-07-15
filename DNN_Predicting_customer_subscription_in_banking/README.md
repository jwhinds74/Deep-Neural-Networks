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

## __Neural Network Modeling Scenarios__
## Proposed Model # 1. Multi-Layer Perceptron (Model_1)
![image](https://github.com/user-attachments/assets/33f91d5a-2984-49c3-b8c9-951c986729aa)

![image](https://github.com/user-attachments/assets/b63647cc-5865-4e6b-b240-64c12cbed5b3)

![image](https://github.com/user-attachments/assets/8fb9a460-ba0b-492b-a8bd-7083fbf5af1c)

![image](https://github.com/user-attachments/assets/dd35dfa4-472d-4625-86ac-7ab536c31822)

![image](https://github.com/user-attachments/assets/74f837f2-ce08-4ef4-b65f-6a94f6383ad2)

## __Proposed Model #2. Multi-Layer Perceptron (Model_2)__
## Original Model - 1st Variant:  Model_2
**Effect of LR on training**

When training a neural network model, we typically use a default schedule with a constant LR to update network weights for each training epoch. With a small LR, the training can progress very slowly, and with a larger LR, we often observe overshooting, that is, an undesirable divergent behavior in our loss function. An optimal LR is a trade-off between the two. To help with this, we use the LR scheduler, which is a framework that makes pre-specified adjustments to the LR at set intervals during the training procedure.

**Learning rate schedule**

Keras provides a LR schedule base class that can be used to adapt the LR of our optimizer during training. This can enable our model to learn good weights early on, and be fine-tuned later.
![image](https://github.com/user-attachments/assets/2ca329f6-69a7-45b5-a048-2683baacc92f)

![image](https://github.com/user-attachments/assets/362d97d1-ac49-4737-9e3b-29c4b3342f94)

![image](https://github.com/user-attachments/assets/4b3419cc-865c-4a41-98d7-fd8506f0837b)

![image](https://github.com/user-attachments/assets/650c5ec2-d75e-4a30-bedf-968512cacb6a)

![image](https://github.com/user-attachments/assets/7b695555-1031-4e83-b003-b73956c79f81)

## __Proposed Model # 3. Feed Forward Neural Network (FFNN Model_3)__
## FFNN Model_3, with Fine-Tune Scenario #1
Let's create, fine-tune, train, and evaluate a Feed Forward Neural Network (FFNN) for this problem. We will use the data that has already been preprocessed and resampled. Here is a structured approach to implement this:

**Model Definition:**

1.- I will define an Feed Forward Neural Network with multiple dense layers, dropout for regularization, and ReLU activations.

2.- I will fine tune with three different model-scenarios focused on Learning Rate based in exponential-decay, step-decay,  and Dropout

![image](https://github.com/user-attachments/assets/7b2ee126-f6ba-4e07-b740-377de350cfae)

![image](https://github.com/user-attachments/assets/46680a47-a3aa-4c54-a197-00f7c355d0bc)

![image](https://github.com/user-attachments/assets/573fe491-7a57-4401-b735-61fd71463fcb)

![image](https://github.com/user-attachments/assets/483f8af2-406f-45ab-8bb7-a23a555ffa22)

![image](https://github.com/user-attachments/assets/a6e72273-d077-47b3-9c30-e0ab0857ac37)

## FFNN Model_3, with Fine-Tune Scenario #2 

![image](https://github.com/user-attachments/assets/05eab3dd-d9cc-4673-8517-18ef54f65dff)

![image](https://github.com/user-attachments/assets/04325d98-e936-4d2e-8f8f-c7ffb6610b9a)

![image](https://github.com/user-attachments/assets/04db87a9-e6cb-457d-adb7-2153b8d04a00)

![image](https://github.com/user-attachments/assets/ce40e1cf-1ebb-43cf-99b4-f2871cbd0c25)

![image](https://github.com/user-attachments/assets/c720aa74-1043-4fd9-b8ff-13d2f3d6db0a)

## __FFNN Model_3, with Fine-Tune scenario #3__

![image](https://github.com/user-attachments/assets/8ab0c9f3-60fa-406e-a6cd-b93b7dcb639b)

![image](https://github.com/user-attachments/assets/963ef3c1-f311-44ea-8f4a-f2b2896fa5f8)

![image](https://github.com/user-attachments/assets/0a0d6d91-f96f-4588-8140-df3172c10529)

![image](https://github.com/user-attachments/assets/d86ad9fc-03c2-4c0d-95df-1f9c3bc33bf4)


## __Decision Based on Performance Analysis__
Based on the performance analysis, the FFNN Model 3 with Fine-Tune Scenario #2 is the best approach. It achieved the highest ROC-AUC score of 0.9241, indicating superior performance in predicting customer subscriptions. This model's fine-tuning with exponential decay learning rate proved to be the most effective strategy for this dataset and business problem.


























