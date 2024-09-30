## DSTS- Assignment1(Predictive Modelling of Eating-Out Problem)

### Project Overview

This project focuses on predictive modeling to analyze restaurant ratings and costs for dining out. The objective is to create regression and classification models to evaluate restaurants in Sydney based on a dataset containing various attributes such as cost, rating, cuisine, and location. Additionally, the project includes a Tableau dashboard and Dockerized model for deployment.

### Key Features
Regression Models: Used to predict restaurant costs based on multiple factors.
Classification Models: Classify restaurants based on user ratings.
Tableau Visualization: Interactive dashboard to visualize restaurant data.
Dockerized Models: Models are containerized for easy deployment.
Git Workflow: Version control through GitLab and GitHub.

## Setup Instructions
- Prerequisites
- Python 3.x
- Docker
- Jupyter Notebook
- Git

## Installation

Clone the repository:

https://github.com/Jigme96-BHU/dsts-assignment1.git

## Install the required packages:

```bash
pip install -r requirements.txt

```

## Run the Jupyter notebook:

```bash
jupyter notebook assignment1.ipynb
```

## Tableau Dashboard

The interactive visualization for this project is available on Tableau:

Link: https://public.tableau.com/views/dsts-assignment1/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

## Dockerizing the Models

#### run docker models

```bash
docker run jigme26/dsts-assignment:latest
```

## Exected Output
- **Data Exploration**: You will see data visualizations and insights regarding the dataset, such as histograms and Bar plots.
- **Regression Models**: Running the regression script will produce metrics like:
  - MSE - mean score error (a common metric used to evaluate the performance of regression models) 
- **Classification Models**: After running the ipynb script, the classification output will include:
  - Model accuracy(It represents the proportion of correctly classified instances out of the total number of instances.)
  - Confusion matrixa (a table that is used to describe the performance of a classification model.)
- **Tableau Visualizations**: If using `dsts-assignment1.twb` or click the link, you'll get interactive visualizations related to Sydney's restaurants data.

- **docker run jigme26/assignment1**:

```bash
----------------------------------------------------------------------
Mean Squared Error for Linear Regression: 0.19354723644735014
Mean Squared Error for Sgd Regression: 0.1941190177703764


-----------------------------------------------------------------------
Accuracy for Log Classification: 0.8005974607916355
Confusion Matrix for Log Classification:
[[752  86]
 [181 320]]
Accuracy for Random forest Classification: 0.8416728902165795
Confusion Matrix for Random forest Classification:
[[728 110]
 [102 399]]
Accuracy for Mlp Classification: 0.7251680358476476
Confusion Matrix for Mlp Classification:
[[745  93]
 [275 226]]
Accuracy for Decision tree Classification: 0.8125466766243465
Confusion Matrix for Decision tree Classification:
[[716 122]
 [129 372]]

```