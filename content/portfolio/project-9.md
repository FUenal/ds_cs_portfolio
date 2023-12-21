---
categories:
- cybersecurity
- data science
date: "2023-12-21T12:14:34+06:00"
description: Juyper Notebook.
draft: false
github_url: '[Open Github](https://github.com/FUenal/deep-learning-cybersecurity-ids/tree/main)'
image: images/portfolio/fastai_vs_xgboost.png
project_url: '[View Jupyter Notebook](https://colab.research.google.com/drive/1WF-cAHsbrZzvpLabhDhQLEj3wB6PmERq?usp=sharing)'
title: Deep learning vs. XGBoost on network traffic data
---


### Project Details

I use the [fast.ai](https://www.fast.ai/) deep learning framework for one of its newest applications: classification on tabular data. I compare its performance against the incumbent best tool in the field, gradient boosting with [XGBoost](https://xgboost.readthedocs.io/en/latest/), as well as against various scikit-learn classifiers in detecting network intrusion traffic and classifying common network attack types (e.g., FTP-BruteForce, DOS-GoldenEye, BruteForce-XSS, SQL-Injection, Infiltration, BotAttack). In line with recent prominence on other tabular datasets, fast.ai is on par with XGBoost and sklearn’s Random Forest Classifier, demonstrating high accuracy (up to 99%), with low false positive and negative rates in the classification of various intrusion types. *Pretty powerful!*

![plot](/images/portfolio/result02032018_plt.png)

### Background

Recent advancements in deep learning algorithms have facilitated significant strides in addressing challenging computer science problems and applications in nearly all areas of life. These breakthroughs have extended to areas such as computer vision, natural language processing, complex reasoning tasks like playing board games (e.g., Go, Chess), and even surpassing human champions. 

In light of the ongoing surge in cyber-attacks and the increased demand for AI usage in the context of cybersecurity [MIT Report](https://wp.technologyreview.com/wp-content/uploads/2022/07/Deep-Learning-Delivers-proactive-Cyber-defense-FNL.pdf), in this project, I investigate the effectiveness and capacity of a powerful new deep learning algorithm, fast ai, in the domain of network intrusion detection and compare its performance against the incumbent best tool in the field, gradient boosting with XGBoost, as well as against various scikit-learn classifiers (random forest, knn, naïve bayes, etc.). 

In a previous study, [Basnet and colleagues (2018)]( https://isyou.info/jisis/vol9/no4/jisis-2019-vol9-no4-01.pdf) have shown that the fastai deep learning algorithm provided the highest accuracy of about 99% compared to other well-known deep learning frameworks (e.g., Keras, TensorFlow, Theano) in detecting network intrusion traffic and classifying common network attack types using the [CSE-CIC-IDS2018 dataset](https://www.unb.ca/cic/datasets/ids-2018.html) (same dataset as I used here). 

Deep learning is the gold standard for large, unstructured datasets, including text, images, and video and has been battle tested in areas such as computer vision, natural language processing, and complex reasoning tasks. However, for one specific type of dataset –one of the most common datasets used in cybersecurity– deep learning typically falls behind other, more “shallow-learning” approaches such as decision tree algorithms (random forests, gradient boosted decision trees): TABULAR DATA. 

Indeed, in a [systematic review and meta-analysis](https://arxiv.org/abs/2207.08815) last year, Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux have shown that, overall, tree-based models (random forests and XGBoost) outperform deep learning methods for tabular data on medium-sized datasets (10k training examples). However, the gap between tree-based models and deep learning becomes narrower as the dataset size increases (here: 10k -> 50k).

Here, I extend these lines of investigation, by comparing fast ai’s deep learning framework with XGBoost as well as other scikit-learn classifiers on a relatively large dataset of network traffic data. Given the large size of the used datasets in this project, I expect fast.ai to achieve comparable results to the other algorithms.

### Dataset

I use the open source CSE-CIC-IDS2018 dataset, a contemporary network intrusion dataset produced and released in 2018(1). 

Further details about the datasets, including the experiments and testbeds utilized for dataset generation, can be found following this [Link](https://www.unb.ca/cic/datasets/ids-2018.html). The datasets comprise both benign (normal) network traffic and malicious traffic resulting from various network attacks, briefly outlined below. Table 1 provides an overview of the number of samples and network traffic types in each dataset.

**Table 1: Number of samples and network traffic types in each dataset**

| File Name      | Traffic Type     | # Samples | # Dropped |
| :------------- | :--------------: | --------: | --------: |
| 02-14-2018.csv | Benign           |   663,808 | 3818      |
|                | FTP-BruteForce   |   193,354 | 6         |
|                | SSH-Bruteforce   |   187,589 | 0         |
| -------------- | ---------------  | --------- | --------- |
| 02-15-2018.csv | Benign           |   988,050 | 8027      |
|                | DOS-GoldenEye    |    41,508 | 0         |
|                | DOS-Slowloris    |    10,990 | 0         |
| -------------- | ---------------  | --------- | --------- |
| 02-16-2018.csv | Benign           |   446,772 | 0         |
|                | Dos-SlowHTTPTest |   139,890 | 0         |
|                | DoS-Hulk         |   461,912 | 0         |
| -------------- | ---------------  | --------- | --------- |
| 02-22-2018.csv | Benign           | 1,042,603 | 5610      |
|                | BruteForce-Web   |       249 | 0         |
|                | BruteForce-XSS   |        79 | 0         |
|                | SQL-Injection    |        34 | 0         |
| -------------- | ---------------  | --------- | --------- |
| 02-23-2018.csv | Benign           | 1,042,301 | 5708      |
|                | BruteForce-Web   |       362 | 0         |
|                | BruteForce-XSS   |       151 | 0         |
|                | SQL-Injection    |        53 | 0         |
| -------------- | ---------------  | --------- | --------- |
| 03-01-2018.csv | Benign           |   235,778 | 2259      |
|                | Infiltration     |    92,403 | 660       |
| -------------- | ---------------  | --------- | --------- |
| 03-02-2018.csv | Benign           |   758,334 | 4050      |
|                | BotAttack        |   286,191 | 0         |


### Data cleaning
Extensive data cleaning and feature engineering were needed to process the data for the over 6 million entries in this dataset. Following the same approach as Basnet and colleagues, after downloading the dataset, I conducted an analysis of its characteristics and performed necessary cleaning procedures. The dataset comprises original traffic stored in pcap files, logs, as well as preprocessed, labeled, and feature-selected CSV files. My focus was on the labeled CSV files, encompassing a total of 80 traffic features extracted using the CICFlowMeter. This dataset encompasses both normal (benign) and attack traffic, featuring various common attack types outlined in the preceding section, distributed across seven CSV files.

To streamline my experiments and due to an abundance of available samples, I decided to simplify by discarding instances with Infinity, NaN, or missing values. Additionally, I converted timestamps to Unix epoch numeric values and addressed issues such as parsing and removing repeated column headers in certain data files. Approximately 20,000 samples were eliminated during the data cleanup process. Table 1 provides a summary of the datasets utilized in the experiments post the data cleanup step. The "Number of Samples Remaining" column denotes the total samples retained after the removal of instances from each category, as outlined in the last column. Each dataset contains a specific number of traffic samples belonging to benign and one or more attack types, repeated across multiple datasets, as summarized in Tables 1 and 2.

**Table 2: Total number of traffic data samples for each type among all the datasets**

| Traffic Type     | # Samples |
| ---------------- | --------: |
| Benign           | 5,177,646 |
| FTP-BruteForce   |   193,354 |
| SSH-BruteForce   |   187,589 |
| DOS-GoldenEye    |    41,508 |
| Dos-Slowloris    |    10,990 |
| Dos-SlowHTTPTest |   139,890 |
| Dos-Hulk         |   461,912 |
| BruteForce-Web   |       611 |
| BruteForce-XSS   |       230 |
| SQL-Injection    |        87 |
| Infiltration     |    92,403 |
| BotAttack        |   286,191 |
| Total Attack     | 1,414,765 |


### Tabular learning with fast.ai

Here, I am providing a walk-through using one of the seven datasets ("03-02-2018.csv") available in the CSE-CIC-IDS2018 dataset. An overview of the overall results including all datasets is available in the [Github Repo](https://github.com/FUenal/deep-learning-cybersecurity-ids/tree/main). First, I start by using the fast.ai deep learning framework. 
The tabular feature in fast.ai consolidates training, validation, and, if desired, testing data into a unified TabularPandas object. This arrangement allows for refining pre-processing steps on the training data, subsequently applying them consistently to the validation and test data. As a result, with fast.ai, the tasks of normalizing, handling missing values, and identifying categories for each categorical variable are predominantly automated. Furthermore, as demonstrated later on, this processed data can be employed to train models from alternative libraries.

The first code chunk below, takes in the cleaned data, pre-processes it, build dataloaders and a learner, learns from the data, fits the model, and finally tests the predictions on the test dataset

```python
# Pre-process data

from fastai.tabular.all import *

path = '/content/sample_data/results.csv' # use your path

dls = TabularDataLoaders.from_csv('results.csv', path=path, y_names="Label",
    cat_names = cat_names,
    cont_names = cont_names,
    procs = [Categorify, FillMissing, Normalize])
    
# Split data into training and validation sets
splits = RandomSplitter(valid_pct=0.2)(range_of(data_cleaned))

# Create tabularpandas
to = TabularPandas(data_cleaned, procs=[Categorify, FillMissing,Normalize],
                   cat_names = cat_names,
                   cont_names = cont_names,
                   y_names='Label',
                   splits=splits)
                   
# Build dataloaders
dls = to.dataloaders(bs=256)

# Define learner
learn = tabular_learner(dls, metrics=accuracy, layers=[200,100])

# Fit model
learn.fit_one_cycle(3)


# Determine best learning rate. The lr_find() function in fast.ai allows one to see the loss that different learning rates would cause. 
learn.lr_find()

# Generate test set
test_df = data_cleaned.copy()
test_df.drop(['Label'], axis=1, inplace=True)
dl = learn.dls.test_dl(test_df)

# Get preditction on test set
learn.get_preds(dl=dl)

# load dataloaders on testset
dls = to.dataloaders()

# Build learner
learn = tabular_learner(dls, layers=[200,100], metrics=accuracy)

# Fit model on testset
learn.fit(3, 1e-2)

# Now we'll grab predictions
nn_preds = learn.get_preds()[0]
nn_preds

```

Let's take a look at the confusion matrix.

```python

# Confusion matrix
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

```

![confusion_matrix](/images/portfolio/result02032018_cm.png)

The confusion matrix shows extremely low false positive and false negative scores indicating good recall and precision.

### XGBoost and other models

In the second step, we take the data which we pre-processed using fast.ai and feed it into scikit-learn and train different models to compare the accuracy of the fast.ai deep learning framework with other machine learning algorithms.

```python
# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# load dataset
X, y = to.train.xs, to.train.ys.values.ravel()

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=500)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('BernoulliNB', BernoulliNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('XGB', XGBClassifier(objective = 'multi:softmax', booster = 'gbtree', 
                     nrounds = 'min.error.idx', num_class = 3, 
                     maximize = False, eval_metric = 'logloss', eta = .1,
                     max_depth = 14, colsample_bytree = .4, n_jobs=-1)))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)
	print(msg)
	
```

```python
# Create results dataframe
result02032018 = pd.DataFrame(columns=["Classifier", "Accuracy", "Standard Deviation"])

result02032018 = result02032018.append(pd.DataFrame([['Logistic Regression', 97.04, 1.55]], columns=["Classifier", "Accuracy", "Standard Deviation"]))
result02032018 = result02032018.append(pd.DataFrame([['LinearDiscriminantAnalysis', 94.41, 0.08]], columns=["Classifier", "Accuracy", "Standard Deviation"]))
result02032018 = result02032018.append(pd.DataFrame([['KNN', 99.99, 0.00]], columns=["Classifier", "Accuracy", "Standard Deviation"]))
result02032018 = result02032018.append(pd.DataFrame([['DecisionTreeClassifier', 99.99, 0.00]], columns=["Classifier", "Accuracy", "Standard Deviation"]))
result02032018 = result02032018.append(pd.DataFrame([['GaussianNB', 83.98, 0.13]], columns=["Classifier", "Accuracy", "Standard Deviation"]))
result02032018 = result02032018.append(pd.DataFrame([['BernoulliNB', 94.11, 0.08]], columns=["Classifier", "Accuracy", "Standard Deviation"]))
result02032018 = result02032018.append(pd.DataFrame([['RandomForestClassifier', 99.99, 0.00]], columns=["Classifier", "Accuracy", "Standard Deviation"]))
result02032018 = result02032018.append(pd.DataFrame([['XGBClassifier', 99.99, 0.00]], columns=["Classifier", "Accuracy", "Standard Deviation"]))
result02032018 = result02032018.append(pd.DataFrame([['fast.ai', 99.99, 0.00]], columns=["Classifier", "Accuracy", "Standard Deviation"]))

result02032018 = result02032018.sort_values('Accuracy', ascending=True)

# Plot algorithm comparison
fig, ax = plt.subplots(1,1, figsize=(6,5))

bars = ax.barh(result02032018['Classifier'], result02032018['Accuracy'], color="b")
ax.bar_label(bars)
ax.tick_params(axis="y", labelsize=14)
ax.set_xlabel('Accuracy %', fontsize=14)
ax.set_title('Classifier Accuracy', fontsize=20);
```

Now let's take a look at the comparison of accuracy scores between the different approaches.

![plot](/images/portfolio/result02032018_plt.png)

### Results

For the dataset that I present here ("03-02-2018.csv"), as well as the majority of datasets available, fast.ai is on par with XGBoost and RandomForestClassifier. All of the models have been trained using default measures, so using parameter fine-tuning might still improve the results. However, given the almost perfect accuracy score, fine-tuning seemed a little too much. Overall, except for one dataset ("03-01-2018.csv"), fast.ai achieved high accuracy scores and could compete with the more battle-tested ML approaches, which have been the go to tools when it comes to tabular data. All frameworks fare well, irrespective of severe class-imbalance in some of the datasets.


#### Project Requirements

✅ Python

✅ Pandas

✅ Numpy

✅ Matplotlib

✅ fast.ai

✅ scikit-learn

✅ xgboost



1- A. L. I. Sharafaldin and A. Ghorbani. Toward generating a new intrusion detection dataset and intrusion traffic characterization. In Proc. of the 4th International Conference on Information Systems Security and Privacy (ICISSP’18), Funchal, Madeira, Portugal, pages 108–116. ICISSP, January 2018.

