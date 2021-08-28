# A-complete-tour-of-Decision-Trees-and-Ensemble-Methods

In this blog we will be seeing decision trees and several ensemble methods and use cases of all of them in detail.

Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. It follows a set of if-else conditions to visualize the data and classify it according to the conditions. Remember, decision trees are non-linear classifiers.

A tree can be “_learned_” by splitting the source set into subsets based on an attribute value test. This process is repeated on each derived subset in a recursive manner called recursive partitioning. Growing a tree involves deciding on which features to choose and what conditions to use for splitting, along with knowing when to stop. We choose the feature and value by using metrics such as gini impurity and information gain for classification and MSE or MAE for regression. These works as a cost function. Both combined are called as **CART**(Classification And Regression Tree.

Decision tree work well when input values are categorical instead of continuous.

Geometrically we can think **_decision trees as a set of a number of axis parallel hyperplanes which divides the space into the number of hypercuboids during the inference._** we classify the point based on which hypercuboid it falls into.

![image](https://user-images.githubusercontent.com/65160713/131227877-9967622d-6e77-40ef-953e-940bb7d780af.png)

## Decision Trees for Classification-

### GINI Impurity-

The measure of the degree of probability of a particular variable being wrongly classified when it is randomly chosen is called the Gini index or Gini impurity. It is a metric to measure how often a randomly chosen element would be incorrectly identified. The data is equally distributed based on the Gini index.

Mathematical Formula :

![image](https://user-images.githubusercontent.com/65160713/131227902-02cf367a-2062-4fba-8c32-a45c059f9eb4.png)

_pᵢ = probability of an object being classified into a particular class._

Generally we do binary split of yes or no condition. In that case it will be-

      Gini=1- p(yes)² -p(no)²

When you use the Gini index as the criterion for the algorithm to select the feature for the root node. The feature with the least Gini index is selected. The value of Gini Impurity lies between 0 and 1.

When we do classification with CART algorithm then we use gini impurity as a splitting metric.

### Entropy- 

Entropy is the main concept of this algorithm, which helps determine a feature or attribute that gives maximum information about a class is called Information gain or ID3 algorithm. By using this method, we can reduce the level of entropy from the root node to the leaf node. Information gain computes the difference between entropy before split and average entropy after split of the dataset based on given attribute values.

Entropy is nothing but the measure of disorder. The Mathematical formula for Entropy is as follows -

![image](https://user-images.githubusercontent.com/65160713/131227987-5a6c4161-9a95-47b8-a785-c0700fead3e4.png)

This measure satisfies our criteria, because of the -p*log2(p) construction: when p gets close to zero (i.e., the category has only a few examples in it), then the log(p) becomes a big negative number, but the p part dominates the calculation, so the entropy works out to be nearly zero. Remembering that entropy calculates the disorder in the data, this low score is good, as it reflects our desire to reward categories with few examples in. Similarly, if p gets close to 1 (i.e., the category has most of the examples in), then the log(p) part gets very close to zero, and again ‘p’ which dominates the calculation, so the overall value gets close to zero. Hence we see that when both the category is nearly or completely empty, or when the category nearly contains or completely contains all the examples, the score for the category gets close to zero, which models what we wanted it to. And this is what we want because for example take a coin. Maximum entropy will be when coin will be unbiased as then there will be equal probability for both sides so result will be total random and we can’t predict outcome resulting in maximum entropy. But if coin will be biased for one side, say heads side, then probability of heads will be more and we can predict head as outcome. So, result will not be total random and hence entropy will be low then. In this sense, we can also call entropy as measure of randomness. Note that 0*ln(0) is taken to be zero by convention.

Lower the value of entropy, higher is the purity of the node. The entropy of a homogeneous node is zero. Entropy is measured between 0 and 1.(Depending on the number of classes in your dataset, entropy can be greater than 1 but it means the same thing , a very high level of disorder).

### Information gain(ID3 (Iterative Dichotomiser 3))-

Now we know how to measure disorder. Next we need a metric to measure the reduction of this disorder in our target variable/class given additional information( features/independent variables) about it. This is where Information Gain comes in. Mathematically it can be written as:

![image](https://user-images.githubusercontent.com/65160713/131228012-ed270601-b9ba-467d-b76a-e596c27fea19.png)

_Information Gain from X on Y_

We simply subtract the entropy of Y given X from the entropy of just Y to calculate the reduction of uncertainty about Y given an additional piece of information X about Y. This is called Information Gain. The greater the reduction in this uncertainty, the more information is gained about Y from X.

The feature or attribute with the highest ID3 gain is used as the root for the splitting.

Information gain (IG) is biased toward variables with large number of distinct values not variables that have observations with large values. Suppose a date variable is there in dataset. Now, let’s calculate the Information Gain for “Date”. We can start to calculate the entropy for one of the dates, such as “2020–01–01”.

![image](https://user-images.githubusercontent.com/65160713/131228033-c1f96f21-dbfb-4b7f-aa61-1ac8e9ba3134.png)

Since there is only 1 row for each date, the final decision must be either “Yes” or “No”. So, the entropy must be 0! In terms of the information theory, it is equivalent to say:

_The date tells us nothing, because the result is just one, which is certain. So, there is no “uncertainty” at all._

Similarly, for all the other dates, their entropies are 0, too.

Now, let’s calculate the entropy for the date itself.

![image](https://user-images.githubusercontent.com/65160713/131228052-adbbf66e-a686-4bee-9239-81a435b7cc63.png)

WoW, that is a pretty large number compared to the other features. And so, our algorithm will use date as a splitting criteria. But we know that date doesn’t tell anything about the data. So, this is a disadvantage of ID3.

_[Note- Some people consider information gain and ID3 two different things. They consider information gain can use both gini impurity and entropy as it’s measure, making it’s formula as-
IG(Y,X)=I(Y)-I(Y|X)
where I is impurity criterion and it can be gini impurity or entropy. But when we use entropy as impurity criterion, then it is called as ID3 algorithm. So, not to be confused as they may be used interchangeable. But widely accepted one is the one which treats information gain and ID3 same means which we discussed previously.]_

Remember ID3 algorithm can’t handle continuous attributes.

### Information Gain vs Gini Index-

Both gini and entropy are measures of impurity of a node. But if we compare both the methods then Gini Impurity is more efficient than entropy in terms of computing power. As entropy also have to calculate log term, it takes more time. Gini’s maximum impurity is 0.5 and maximum purity is 0. Entropy’s maximum impurity is 1 and maximum purity is 0. Generally, your performance will not change whether you use Gini impurity or Entropy. Infact, according to a research, It only matters in 2% of the cases whether you use gini impurity or entropy. So, the only main difference is of computing power.

![image](https://user-images.githubusercontent.com/65160713/131228079-669b4af7-cf1e-477e-87c8-2333b71f6a76.png)

**Q.) Why we don’t use directly the accuracy as feature selection measure? The feature on split which gives more accuracy , why not to split on that?**
**Ans.** Decision trees are generally prone to over-fitting and accuracy doesn’t generalize well to unseen data. The above discussed approaches are usually more stable and also chooses the most impactful features close to the root of the tree.

### C4.5 algorithm-

It is just an extension of ID3 algorithm. C4.5 made a number of improvements to ID3. Some of these are:

1. Handling both continuous and discrete attributes — In order to handle continuous attributes, C4.5 first sorts the data and then creates a threshold and then splits the list into those whose attribute value is above the threshold and those that are less than or equal to it. The trick there is to sort your data by the continuous variable in ascending order. Then iterate over the data picking a threshold between data members.
2. Handling training data with missing attribute values — C4.5 allows attribute values to be marked as ? for missing. Missing attribute values are simply not used in gain and entropy calculations.
3. Handling attributes with differing costs.
4. Pruning trees after creation — C4.5 goes back through the tree once it’s been created and attempts to remove branches that do not help by replacing them with leaf nodes.
5. As there was a limitation of information gain that it tends to use the feature that has more unique values. C4.5 solves this problem by using information gain ratio instead of information gain. Information Gain Ratio is simply adding a penalty on the Information Gain by dividing with the entropy of the parent node.

![image](https://user-images.githubusercontent.com/65160713/131228098-163c67d6-a0a1-476b-bf7d-188df19a3615.png)

It doesn’t solve this problem all time but still Information Gain Ratio will be quite enough to avoid most of the scenarios that Information Gain will cause bias.

### C5 algorithm-

C5 is an advanced version of C4.5 algorithm. It is generally more accurate, much fast and consumes less memory. C5.0 gets similar results to C4.5 with considerably smaller decision trees. As it is much faster than C4.5, then it can also handle large datasets with ease. It supports boosting too which improves the tree and gives more accuracy. But it is rarely used as none of the library have implemented it. So for using it you have to implement it from scratch.

### CART algorithm-

CART(Classification And Regression Trees) is used by sklearn decision trees. scikit-learn uses an optimized version of the CART algorithm; however, scikit-learn implementation does not support categorical variables for now. So we encode the categorical variables. CART is much same as c4.5 with some differences.

ID3, C4.5, C5 can only perform classification but CART can also perform regression too. It does that by changing gini impurity to variance.

CART prunes trees using a cost-complexity model whose parameters are estimated by cross-validation while C4.5 uses a single-pass algorithm derived from binomial confidence limits.

Rather than general trees that could have multiple branches which are generated by ID3 , c4.5 and C5, CART makes use binary tree, which has only two branches from each node. It uses gini index as a metric for splitting for classification.

It’s on your choice that how you choose threshold for continuous variables. Some check purity on every data point and choose the best as threshold. But it takes a lot of time so some people sort the continuous variables and check for some values like values after leaving 2 points or leaving 3 points. It all depends on choice of coder. But if you have high performance gpu then it may not take much time to examine it for every value.

### CART vs ID3 vs C4.5-

![image](https://user-images.githubusercontent.com/65160713/131228138-2c318312-6158-40a3-9aaf-52a5ea34be34.png)

### ID3-

#### Advantages-

• Understandable prediction rules are created from the training data.

• Builds the fastest tree.

• Builds a short tree.

• Only need to test enough attributes until all data is classified. 

• Finding leaf nodes enables test data to be pruned, reducing number of tests. 

• Whole dataset is searched to create tree.

#### Disadvantages-

• Data may be over-fitted or over-classified, if a small sample is tested. 

• Only one attribute at a time is tested for making a decision.

• Does not handle numeric attributes and missing values.

### C4.5-

#### Advantages-

• Can handle continuous features too

• Can handle missing values

• Also does pruning to remove weak nodes.

#### Disadvantages-

• C4.5 constructs empty branches; it is the most crucial step for rule generation in C4.5.We have found many nodes with zero values or close to zero values. These values neither contribute to generate rules nor help to construct any class for classification task. Rather it makes the tree bigger and more complex

• Over fitting happens when algorithm model picks up data with uncommon characteristics.

• Susceptible to noise more than CART and ID3.

### CART-

#### Advantages-

• CART can easily handle both numerical and categorical variables.

• CART algorithm will itself identify the most significant variables and eliminate nonsignificant ones.

• CART can easily handle outliers.

• Most libraries use CART.

#### Disadvantages-

• CART may have unstable decision tree. Insignificant modification of learning sample such as eliminating several observations and cause changes in decision tree: increase or decrease of tree complexity, changes in splitting variables and values.

• CART splits only by one variable.

CART does binary splits. ID3, C45 and the family exhaust one attribute once it is used. This makes sometimes a difference which means that in CART the decisions on how to split values based on an attribute are delayed. Which means that there are pretty good chances that a CART might catch better splits than C45. The drawback is that with CART you can’t create rules and the whole tree is larger and harder to interpret. Anyway, the interpretation is not always useful.

### Pruning-

Pruning reduces the complexity of the final classifier, and hence improves predictive accuracy by the reduction of overfitting. Decision trees are the most susceptible out of all the machine learning algorithms to overfitting and effective pruning can reduce this likelihood. A tree that is too large risks overfitting the training data and poorly generalizing to new samples. A small tree might not capture important structural information about the sample space. However, it is hard to tell when a tree algorithm should stop because it is impossible to tell if the addition of a single extra node will dramatically decrease error. This problem is known as the horizon effect. A common strategy is to grow the tree until each node contains a small number of instances then use pruning to remove nodes that do not provide additional information. Pruning should reduce the size of a learning tree without reducing predictive accuracy.

### Pre-Pruning-

Pre-pruning procedures prevent a complete induction of the training set by replacing a stop criterion in the induction algorithm (e.g. max. Tree depth or information gain (Attr)> minGain). Pre-pruning methods are considered to be more efficient because they do not induce an entire set, but rather trees remain small from the start. Prepruning methods share a common problem, the horizon effect. This is to be understood as the undesired premature termination of the induction by the stop () criterion. Early stopping may underfit by stopping too early. The current split may be of little benefit, but having made it, subsequent splits more significantly reduce the error.

### Post-Pruning-

Post-pruning (or just pruning) is the most common way of simplifying trees. Here, nodes and subtrees are replaced with leaves to improve complexity. Pruning can not only significantly reduce the size but also improve the classification accuracy of unseen objects. It may be the case that the accuracy of the assignment on the test set deteriorates, but the accuracy of the classification properties of the tree increases overall.

**Minimal Cost-Complexity Pruning** is one of the types of Pruning of Decision Trees. This algorithm is parameterized by α(≥0) known as the complexity parameter. The complexity parameter is used to define the cost-complexity measure, Rα(T) of a given tree T:
      Rα(T)=R(T)+α|T|

_where |T| is the number of terminal nodes in T and R(T) is traditionally defined as the total misclassification rate of the terminal nodes._

In its 0.22 version, Scikit-learn introduced this parameter called ccp_alpha (Yes! It’s short for **Cost Complexity Pruning- Alpha**) to Decision Trees which can be used to perform the same.

Decision Tree in sklearn has a function called **cost_complexity_pruning_path**, which gives the effective alphas of subtrees during pruning and also the corresponding impurities. In other words, we can use these values of alpha to prune our decision tree:

![image](https://user-images.githubusercontent.com/65160713/131228364-6301ff18-b64a-4281-9554-0f50c2c19a8d.png)

We will set these values of alpha and pass it to the ccp_alpha parameter of our DecisionTreeClassifier. By looping over the alphas array, we will find the accuracy on both Train and Test parts of our dataset. Then we choose the best value.

### Categorical features for decision trees in sklearn-

Sklearn’s decision tree don’t work for categorical features. It treats categorical features too as numeric features. Like if there is a feature with category 0,1 and 2, then it may split the node like if it is greater than 0.7 or not or 1.3 or not. But we need to do one hot encoding of categorical features. The problem with coding categorical variables as integers is that it imposes an order on them, which may or may not be meaningful, depending on the case; for example, you could encode _['low', 'medium', 'high']_ as _[0, 1, 2_], since 'low' < 'medium' < 'high' (we call these categorical variables ordinal), although you are still implicitly making the additional (and possibly undesired) assumption that the distance between 'low' and 'medium' is the same with the distance between 'medium' and 'high' (of no impact in decision trees, but of importance e.g. in k-nn and clustering). But this approach fails completely in cases like, say, [_'red','green','blue'_] or [_'male','female'_], since we cannot claim any meaningful relative order between them. Hence, we do encoding of feature so our model treats every value equally.

If a continuous variable is chosen for a split, then there would be a number of choices of values on which a tree can split and in most cases, the tree can grow in both directions. Categorical variables are naturally disadvantaged in this case and have only a few options for splitting which results in very sparse decision trees. The situation gets worse in variables that have a small number of levels and one-hot encoding falls in this category with just two levels. The trees generally tend to grow in one direction because at every split of a categorical variable there are only two values (0 or 1). The tree grows in the direction of zeroes in the dummy variables. If we have a categorical variable with q levels, the tree has to choose from ((2^q/2)-1) splits. For a dummy variable, there is only one possible split and this induces sparsity.

_One-hot encoding categorical variables with high cardinality can cause inefficiency in tree-based ensembles. Continuous variables will be given more importance than the dummy variables by the algorithm which will obscure the order of feature importance resulting in poorer performance. So we use other encodings for better result._

We use-

•  Categorical features with large cardinalities (over 1000): Binary encoding

• Categorical features with small cardinalities (less than 1000): Numeric encoding

## Decision Trees for Regression-

We generally use CART for building regression trees by replacing gini impurity by other metrics such as variance reduction, MSE or MAE. Mostly we use MSE. We can also use ID3 and C4.5 and replace entropy by other metrics to do regression. For regression trees, the value of terminal nodes is the mean of the observations falling in that region. Therefore, if an unseen data point falls in that region, we predict using the mean value.

![image](https://user-images.githubusercontent.com/65160713/131228724-2874ba0d-053d-4118-82b7-9d0e287e7c8f.png)

### MSE-

The first step to create a tree is to create the first binary decision. How are you going to do it?

• We need to pick a variable and the value to split on such that the two groups are as different from each other as possible.

• For each variable, for each possible value of the possible value of that variable see whether it is better.

• How to determine if it is better? Take weighted average of two new nodes (mse*num_samples)

[_A way to find the best split which is to try every variable and to try every possible value of that variable and see which variable and which value gives us a split with the best score._]

### Reduction in variance-

Variance is used for calculating the homogeneity of a node. If a node is entirely homogeneous, then the variance is zero.

![image](https://user-images.githubusercontent.com/65160713/131228750-fc5eddda-ef77-4954-aebf-5821ed8ac21e.png)

Here are the steps to split a decision tree using reduction in variance:

1. For each split, individually calculate the variance of each child node
2. Calculate the variance of each split as the weighted average variance of child nodes
3. Select the split with the lowest variance as lower value of variance leads to more pure nodes as all values are nearly same.
4. Perform steps 1–3 until completely homogeneous nodes are achieved

**Are there circumstances when it is better to split into 3 groups ?**

It is never necessary to do more than one split at a level because you can just split them again.

**The MSE at each node it is calculated for which underlying model?**
The underlying model is simply the average of the data points. For the initial root mode is what if we just predicted the average of the dependent variable of all our training data points. Another possible option would be instead of using the average to use median or we can even run a linear regression model. There are a lot of things we could do but in practice the average works really well. There do exist random forests models where the leaf nodes are independent linear regressions but they’re not widely used.

In the kind of problems where any tree based algorithm is useless neural net or linear regression model are the preferred models. The reason is that you want to use a model that actually has a function or shape that can actually fit something so it can extrapolate nicely.

### Tree-based models Vs Linear models for regression-

1. If the relationship between dependent & independent variable is well approximated by a linear model, linear regression will outperform tree based model.
2. If there is a high non-linearity & complex relationship between dependent & independent variables, a tree model will outperform a classical regression method.
3. If you need to build a model which is easy to explain to people, a decision tree model will always do better than a linear model. Decision tree models are even simpler to interpret than linear regression!

Refer to https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3 and https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680 for tuning sklearn hyperparameters.

## Advantages-

1. Simple to understand, interpret, visualize.
2. Decision trees implicitly perform variable screening or feature selection.
3. **Useful in Data exploration:** Decision tree is one of the fastest way to identify most significant variables and relation between two or more variables.
4. Can handle both numerical and categorical data. Can also handle _multi-output problems._
5. Decision trees require relatively little effort from users for data preparation.
6. Nonlinear relationships between parameters do not affect tree performance.
7. The cost of using the tree (_i.e._, predicting data) is logarithmic in the number of data points used to train the tree.

## Disadvantages-

1. Decision-tree learners can create over-complex trees that do not generalize the data well. This is called overfitting. Also, Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. But these problem can be solved by ensemble methods such as Random Forest or bagging and boosting.
2. Greedy algorithms cannot guarantee to return the globally optimal decision tree. But this problem too can be solved by ensemble methods.
3. **Not fit for continuous variables:** While working with continuous numerical variables, decision tree looses information when it categorizes variables in different categories.
4. **Decision trees can be unstable:** Small variations in the data might result in a completely different tree being generated. This is called variance, which needs to be lowered by methods like _bagging_ and _boosting._
5. Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the data set prior to fitting with the decision tree.
6. Decision trees are prone to errors in classification problems with many class and relatively small number of training examples.
7. Decision tree can be computationally expensive to train. The process of growing a decision tree is computationally expensive. At each node, each candidate splitting field must be sorted before its best split can be found. In some algorithms, combinations of fields are used and a search must be made for optimal combining weights. Pruning algorithms can also be expensive since many candidate sub-trees must be formed and compared.

## Ensemble Methods-

_Ensemble learning helps improve machine learning results by combining several models. It is mainly used for reducing overfitting , removing instability problem and local optimum problem in decision trees._

Ensemble methods are meta-algorithms that combine several machine learning techniques into one predictive model in order to **decrease variance** (bagging),**bias**(boosting), or **improve predictions** (stacking).

Ensemble methods can be divided into two groups:

1. sequential ensemble methods where the base learners are generated sequentially (e.g. AdaBoost).
The basic motivation of sequential methods is to **exploit the dependence between the base learners**. The overall performance can be boosted by weighing previously mislabeled examples with higher weight.
2. parallel ensemble methods where the base learners are generated in parallel (e.g. Random Forest).
The basic motivation of parallel methods is to **exploit independence between the base learners** since the error can be reduced dramatically by averaging.

Most of the time (including in the well known bagging and boosting methods) a single base learning algorithm is used so that we have homogeneous weak learners that are trained in different ways. The ensemble model we obtain is then said to be “homogeneous”. However, there also exist some methods that use different type of base learning algorithms: some heterogeneous weak learners are then combined into an “heterogeneous ensembles model”.

In ensemble learning theory, we call weak learners (or base models) models that can be used as building blocks for designing more complex models by combining several of them. A weak classifier is one that performs better than random guessing, but still performs poorly at designating classes to objects. For example, a weak classifier may predict that everyone above the age of 40 could not run a marathon but people falling below that age could. Now, you might get above 60% accuracy, but you would still be misclassifying a lot of data points! Most of the time, these basics models perform not so well by themselves either because they have a high bias (low degree of freedom models, for example) or because they have too much variance to be robust (high degree of freedom models, for example). Then, the idea of ensemble methods is to try reducing bias and/or variance of such weak learners by combining several of them together in order to create a strong learner (or ensemble model) that achieves better performances.

### Q. Why do we need weak learners?

**Ans.** Because we can’t train many strong learners as it may become very expensive. While weak learners are very fast to train. So it’s main advantage is speed. Also, if you already have a strong learner, the benefits of boosting are less relevant; But when we use weak learners, accuracy increases by a lot. At last, they also helps in avoiding overfitting. If we train several strong learners, they there are high chances for them to overfit. So weak learner also help to reduce variance. Also you don’t use the individual trees, but rather “average” them all together, so for a particular data point (or group of points) the trees that over fit that point (those points) will be average with the under fitting trees and the combined average should neither over or under fit, but should be about right.

One important point is that our choice of weak learners should be coherent with the way we aggregate these models. If we choose base models with low bias but high variance, it should be with an aggregating method that tends to reduce variance whereas if we choose base models with low variance but high bias, it should be with an aggregating method that tends to reduce bias.

### Bootstraping-

Bootstrapping is a sampling technique in which we create subsets of observations from the original dataset, with replacement. The size of the subsets is the same as the size of the original set. We do this because if you create all the models on the same set of data and combine it, will it be useful? There is a high chance that these models will give the same result since they are getting the same input. So, we use different samples to train our model.

### Bagging-
Bagging stands for bootstrap aggregation. One way to reduce the variance of an estimate is to average together multiple estimates. For example, we can train M different trees on different subsets of the data (chosen randomly with replacement) and compute the ensemble:

![image](https://user-images.githubusercontent.com/65160713/131229013-85f28cd9-b2c4-45a0-a9e7-d69c65cf5ed2.png)

Bagging uses bootstrap sampling to obtain the data subsets for training the base learners. For aggregating the outputs of base learners, bagging uses _voting for classification_ and _averaging for regression_.

The recursive nature of picking the samples at random with replacement can improve the accuracy of an unstable machine learning model.

Bagging may not always increase much accuracy because we train several weak learners and combine them and “averaging” weak learners outputs do not change the expected answer but reduce its variance and solve the problem of overfitting. _So, it is mostly used when our priority is decreasing variance instead of decreasing bias._

![image](https://user-images.githubusercontent.com/65160713/131229034-da61e792-f57b-47eb-b004-2cfc78f3ac5e.png)

[**Remember, Bagging is not only restricted to decision trees. It can be used for every other classification or regression model by training models on different samples of data and then taking average for regression and voting for classification of each model’s output for prediction. But it is more widely used in decision trees than other models. This is because decision trees are unstable models. While models like KNN , naive-bayes, SVM, etc are less sensitive to perturbation on training samples and therefore they are called stable learners. And Combining stable learners is less advantageous since the ensemble will not help improve generalization performance.**]

Finally, we can mention that one of the big advantages of bagging is that it can be parallelised. As the different models are fitted independently from each others, intensive parallelisation techniques can be used if required.

### Random forest-

Random Forest Models can be thought of as BAGGing, with a slight tweak. When deciding where to split and how to make decisions, BAGGed Decision Trees have the full disposal of features to choose from. Therefore, although the bootstrapped samples may be slightly different, the data is largely going to break off at the same features throughout each model. In contrary, Random Forest models decide where to split based on a random selection of features. Rather than splitting at similar features at each node throughout, Random Forest models implement a level of differentiation because each tree will split based on different features. As a result, the bias of the forest increases slightly, but due to the averaging of less correlated trees, its variance decreases, resulting in an overall better model. Another advantage of sampling over the features is that **it makes the decision making process more robust to missing data**.

In an extremely randomized trees algorithm randomness goes one step further: the splitting thresholds are randomized. Instead of looking for the most discriminative threshold, thresholds are drawn at random for each candidate feature and the best of these randomly-generated thresholds is picked as the splitting rule. This usually allows reduction of the variance of the model a bit more, at the expense of a slightly greater increase in bias.

### Boosting-

Boosting refers to a family of algorithms that are able to convert weak learners to strong learners. The main principle of boosting is to fit a sequence of weak learners− models that are only slightly better than random guessing, such as small decision trees− to weighted versions of the data. More weight is given to examples that were misclassified by earlier rounds.

The predictions are then combined through a weighted majority vote (classification) or a weighted sum (regression) to produce the final prediction. The principal difference between boosting and the committee methods, such as bagging, is that base learners are trained in sequence on a weighted version of the data.

Being mainly focused at reducing bias, the base models that are often considered for boosting are models with low variance but high bias. For example, if we want to use trees as our base models, we will choose most of the time shallow decision trees with only a few depths. Another important reason that motivates use of low variance but high bias models as weak learners for boosting is that these models are in general less computationally expensive to fit . Indeed, as computations to fit the different models can’t be done in parallel (unlike bagging), it could become too expensive to fit sequentially several complex models.

Boosting will reduce the bias and that’s for sure. It decreases variance too because boosting algorithm takes a weighted average of many weak models, and hence the final model has lower variance than each of the weak models. But it doesn’t guarantee to reduce variance all the time. So, boosting is preferred when our priority is decreasing bias than decreasing variance.

### Adaboost-

1. Initially, all observations in the dataset are given equal weights. The weighted samples always sum to 1, so the value of each individual weight will always lie between 0 and 1. We start with a constant function (no other function can have more bias than a constant function, unless the dataset is so boring that even a constant model fits it) and then go on through a number steps finding a function that has a reasonably low bias.
2. Using this model, predictions are made on the whole dataset.
3. Errors are calculated by comparing the predictions and actual values.
4. While creating the next model, higher weights are given to the data points which were predicted incorrectly.
5. Weights can be determined using the error value. For instance, higher the error more is the weight assigned to the observation.
6. This process is repeated until the error function does not change, or the maximum limit of the number of estimators is reached.

We can use any model as a base learner. Decision tree is by default and more widely accepted.

**Decision Stumps** are like trees in a Random Forest, but not “fully grown.” They have one node and two leaves. AdaBoost uses a forest of such stumps rather than trees. Stumps alone are not a good way to make decisions. A full-grown tree combines the decisions from all variables to predict the target value. A stump, on the other hand, can only use one variable to make a decision.

We calculate the actual influence for this classifier in classifying the data points using the formula:

![image](https://user-images.githubusercontent.com/65160713/131229110-0b2495f2-c4ab-432a-960b-0d40c324af5b.png)

_Alpha_ is how much influence this stump will have in the final classification. Total Error is nothing but the total number of misclassifications for that training set divided by the training set size. Notice that when a Decision Stump does well, or has no misclassifications (a perfect stump!) this results in an error rate of 0 and a relatively large, positive alpha value.

After plugging in the actual values of Total Error for each stump, it’s time for us to update the sample weights which we had initially taken as _1/N_ for every data point. We’ll do this using the following formula:

![image](https://user-images.githubusercontent.com/65160713/131229131-72e2e934-f769-46ba-a95a-da44f63dc268.png)

In other words, the new sample weight will be equal to the old sample weight multiplied by Euler’s number, raised to plus or minus alpha (which we just calculated in the previous step). We then normalize our new weights so they always sum to 1.

The two cases for alpha (positive or negative) indicate:

• Alpha is positive when the predicted and the actual output are same. In this case we decrease the sample weight from what it was before, since we’re already performing well.

• Alpha is negative when the predicted output does not agree with the actual class (i.e. the sample is misclassified). In this case we need to increase the sample weight so that the same misclassification does not repeat in the next stump. This is how the stumps are dependent on their predecessors.

We make a final prediction by taking a “weighted majority vote” for classification and weighted mean as regression.

Notice that there exists variants of the initial Adaboost algorithm such that LogitBoost (classification) or L2Boost (regression) that mainly differ by their choice of loss function.

**AdaBoost is not prone to overfitting.** This can be found out via experiment results, but there is no concrete reason available.

But remember boosting technique learns progressively, it is important to ensure that you have quality data. AdaBoost is also extremely sensitive to Noisy data and outliers so if you do plan to use AdaBoost then it is highly recommended to eliminate them. **AdaBoost has also been proven to be slower than XGBoost.**

### Gradient boosting or GBM(Gradient Boosting Machine)-

Just as we mentioned for adaboost, finding the optimal model under this form is too difficult and an iterative approach is required. The main difference with adaptative boosting is in the definition of the sequential optimisation process. Indeed, gradient boosting casts the problem into a gradient descent one: at each iteration we fit a weak learner to the opposite of the gradient of the current fitting error with respect to the current ensemble model.

One of the very basic assumption of linear regression is that it’s sum of residuals is 0. Although, tree based models are not based on any of such assumptions, but if we think logic (not statistics) behind these assumptions, we might argue that, if sum of residuals is not 0, then most probably there is some pattern in the residuals of our model which can be leveraged to make our model better. So, the intuition behind gradient boosting algorithm is to **leverage the pattern in residuals and strenghten a weak prediction model, until our residuals become randomly (maybe random normal too) distributed.** Once we reach a stage that residuals do not have any pattern that could be modeled, we can stop modeling residuals (otherwise it might lead to overfitting). Algorithmically, we are minimizing our loss function, such that test loss reach it’s minima.

Steps-

1. Initialize the model with a constant value by minimizing the loss function

![image](https://user-images.githubusercontent.com/65160713/131229194-d7deff79-da17-48a7-a7de-c43bbd3414bb.png)

b₀ is the prediction of the model which minimizes the loss function at 0th iteration. So, to minimize it, we take derivative of loss function. For the Squared Error loss, it works out to be the average over all training samples. And for classification log likelihood is used as a loss function generally. But after applying some transformation so it is a function of log(odds) and then taking it’s derivative, it comes to -
      prediction= e * log(odds) / (1 + e * log(odds)) 
      where log(odds)=log(p/1-p).
      
2.  Then we compute residual for all the n samples-

![image](https://user-images.githubusercontent.com/65160713/131229234-1d574fd4-7819-48f2-a6a7-86c63bab3fff.png)

_rᵢₘ is nothing but the derivative of the Loss Function and m is tree number. So, rᵢ₁ denotes that we are talking about building first tree._

We have calculated it before and it results in-

![image](https://user-images.githubusercontent.com/65160713/131229386-ff37f77d-6569-47f5-be85-d7d9ed7d1f5b.png)

where predicted for regression was average over all training samples and for classification was- 

      prediction= e * log(odds) / (1 + e * log(odds)).
Remember, this is the gradient for which gradient boosting includes gradient term.

This is called pseudo residual because if we used different loss function, then we may end with different residual.

3. Now we fit a decision tree on these residuals and then get new predictions from it. In other words, every leaf will contain a prediction as to the value of the residual (not the desired label). We can use other models too but decision tree is more widely preferred. Gradient Boost has a range between **8 leaves to 32 leaves**.

Because of the limit on leaves, one leaf can have multiple values. So for generating a unique output value from a leaf we use-

![image](https://user-images.githubusercontent.com/65160713/131229416-753bd0a9-e395-4318-a4ec-ea3df4ac2023.png)

where γⱼₘ is nothing but the output value of a leaf and j is the leaf number and m is the tree we are building and i is training sample. The summation should be only for those records which goes into making that leaf. So we differentiate it and solve it for minimizing the γ(the term which is at last). For that we put the loss function equal to zero(argmin(L(yᵢ,Fₘ₋₁(xᵢ)+γ))=0). And it results in optmal value of γ. In regression(mean squared error) it results in simply average of leaf’s values. And for classification(log loss), it results in-

![image](https://user-images.githubusercontent.com/65160713/131229484-a107ed5f-948c-4638-ba37-f712636073cd.png)

where PreviousProb are nothing but our previous predictions(Fₘ₋₁(xᵢ)). So we were trying to find the value of γ that when added to the most recent predictions, minimized the loss function.
Now we do prediction for each of our data point by-

![image](https://user-images.githubusercontent.com/65160713/131229529-a7421df6-0c5d-4237-8d58-37e8fbfa600f.png)

We can now calculate new log(odds) prediction and from it a new probability for classification and mean of predictions for regression. This process repeats until we have made the maximum number of trees specified or the residuals get super small.

When our model is built completely, the final prediction will be equal to the constant value by which we initialized, computed in the first step, plus all of the residuals predicted by the trees that make up the forest multiplied by the learning rate.

So, in sumary the algorithm for gradient boost is-

![image](https://user-images.githubusercontent.com/65160713/131229536-75bb0c0b-7735-40a9-a33a-1c4b05c57b63.png)

For more details see videos of statquest with josh starmer gradient boosting.

### Advantages and Disadvantages of Gradient Boost

Advantages of Gradient Boosting are:

1. _Often provides predictive accuracy that cannot be trumped._
2. _**Lots of flexibility —** can optimize on different loss functions and provides several hyper parameter tuning options that make the function fit very flexible._
3. _**No data pre-processing required —** often works great with categorical and numerical values as is._
4. _**Handles missing data —** imputation not required._

Pretty awesome, right? Let us look at some disadvantages too.

1. _Gradient Boosting Models will continue improving to minimize all errors. This can overemphasize outliers and cause overfitting._
2. _Computationally expensive — often require many trees (>1000) which can be time and memory exhaustive.
3. The high flexibility results in many parameters that interact and influence heavily the behavior of the approach (number of iterations, tree depth, regularization parameters, etc.). This requires a large grid search during tuning.
4. Less interpretative in nature, although this is easily addressed with various tools._

### Light GBM- 

Light GBM is a gradient boosting framework that uses tree based learning algorithm. **Light GBM beats all the other algorithms when the dataset is extremely large.** Compared to the other algorithms, Light GBM takes lesser time to run on a huge dataset. Light GBM is a gradient boosting framework that uses tree-based algorithms and follows leaf-wise approach while other algorithms work in a level-wise approach pattern where you find the best possible node to split and you split that one level down. It finds the leaves which will reduce the loss the maximum, and split only that leaf and not bother with the rest of the leaves in the same level. Light GBM is almost 7 times faster than XGBOOST without much difference in accuracy and is a much better approach when dealing with large datasets.

![image](https://user-images.githubusercontent.com/65160713/131229634-2b169446-a381-4aa1-ac7b-08f539e89703.png)

![image](https://user-images.githubusercontent.com/65160713/131229636-6824ae4b-0bda-49cd-b06b-f78a1e4e1bd8.png)

Level-wise training can be seen as a form of regularized training since leaf-wise training can construct any tree that level-wise training can, whereas the opposite does not hold. Therefore, leaf-wise training is more prone to overfitting but is more flexible. This makes it a better choice for large datasets and is the only option available in lightGBM.

It is not advisable to use LGBM on small datasets. Light GBM is sensitive to overfitting and can easily overfit small data. Their is no threshold on the number of rows but it is okay to use it only for data with 10,000+ rows.

It uses two novel techniques: **Gradient-based One Side Sampling** and **Exclusive Feature Bundling (EFB)** which fulfills the limitations of histogram-based algorithm that is primarily used in all GBDT (Gradient Boosting Decision Tree) frameworks.
Both LightGBM and xgboost utilise histogram based split finding in contrast to sklearn which uses GBM ( One of the reasons why it is slow). In histogram based split, we convert our continuous features to discrete bins. **It costs O(data * feature) for histogram building and O(bin * feature) for split point finding.**

![image](https://user-images.githubusercontent.com/65160713/131229650-ab2a1380-bcdc-4184-9124-15a811d5fb47.png)

#### What makes LightGBM special?

LightGBM aims to reduce complexity of histogram building **(O(data * feature))** by down sampling data and feature using **GOSS** and **EFB**. This will bring down the complexity to **(O(data2 * bundles)) where data2 < data and bundles << feature.**

### GOSS-

GOSS (_Gradient Based One Side Sampling_) is a novel sampling method which down samples the instances on basis of gradients. As we know, the gradient (as we calculated in GBM(Step 2-A)) is large if error or residual is large. If error is small then gradient will too be small. So, if error is already small for some data points, then it means model is already good for those data points. A straightforward idea is to discard those data instances with small gradients. However, the data distribution will be changed by doing so, which will hurt the accuracy of the learned model. To avoid this problem, they propose a new method called Gradient-based One-Side Sampling (GOSS). GOSS keeps all the instances with large gradients(large residual errors) and performs random sampling on the instances with small gradients.

Therefore, when down sampling the data instances, in order to retain the accuracy of information gain estimation, should better keep those instances with large gradients (e.g., larger than a pre-defined threshold, or among the top percentiles), and only randomly drop those instances with small gradients.

They prove that such a treatment can lead to a more accurate gain estimation than uniformly random sampling, with the same target sampling rate, especially when the value of information gain has a large range.

The algorithm is pretty straightforward:

1. Keep all the instances with large gradients
2. Perform random sampling on instances with small gradients
3. Introduce a constant multiplier for the data instances with small gradients when computing the information gain in the tree building process.
4. If we select a instances with large gradients and randomly samples b instances with small gradients, we amplify the sampled data by (1-a)/b.

### EFB-

The motivation behind EFB is a common theme between LightGBM and XGBoost. In many real world problems, although there are a lot of features, most of them are really sparse, like one-hot encoded categorical variables. The way LightGBM tackles this problem is slightly different.

The crux of the idea lies in the fact that many of these sparse features are exclusive, i.e. they do not take non-zero values simultaneously. And we can efficiently bundle these features and treat them as one. But finding the optimal feature bundles is an NP-Hard problem.

To this end, the paper proposes a Greedy Approximation to the problem, which is the Exclusive Feature Bundling algorithm. The algorithm is also slightly fuzzy in nature, as it will allow bundling features which are not 100% mutually exclusive, but it tries to maintain the balance between accuracy and efficiency when selecting the bundles.

The algorithm, on a high level, is:

1. Construct a graph with all the features, weighted by the edges which represents the total conflicts between the features
2. Sort the features by their degrees in the graph in descending order
3. Check each feature and either assign it to an existing bundle with a small conflict or create a new bundle.
4. EFB is merging the features to reduce the training complexity. In order to keep the merge reversible we will keep exclusive features reside in different bins.

### Advantages of Light GBM
1. **Faster training speed and higher efficiency**
2. **Lower memory usage:** Replaces continuous values to discrete bins which result in lower memory usage.
3. **Better accuracy than any other boosting algorithm:** It produces much more complex trees by following leaf wise split approach rather than a level-wise approach which is the main factor in achieving higher accuracy. However, it can sometimes lead to overfitting which can be avoided by setting the max_depth parameter.
4. **Compatibility with Large Datasets:** It is capable of performing equally good with large datasets with a significant reduction in training time as compared to XGBOOST.
5. **Parallel learning supported.**
6. 
For using it and hyperparameter tuning refer to- https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc


### XGBoost-

XGBoost stands for _eXtreme Gradient Boosting._

![image](https://user-images.githubusercontent.com/65160713/131229765-b431fdee-9597-4110-86ba-b47326eb290d.png)

XGBoost improves upon the base GBM framework through systems optimization and algorithmic enhancements.

Let’s discuss some features of XGBoost that make it so interesting.

1. **Regularization:** XGBoost has an option to penalize complex models through both L1 and L2 regularization. Regularization helps in preventing overfitting.

2. **Handling sparse data:** Missing values or data processing steps like one-hot encoding make data sparse. XGBoost incorporates a sparsity-aware split finding algorithm to handle different types of sparsity patterns in the data.

3. **Weighted quantile sketch:** Most existing tree based algorithms can find the split points when the data points are of equal weights (using quantile sketch algorithm). However, they are not equipped to handle weighted data. XGBoost has a distributed weighted quantile sketch algorithm to effectively handle weighted data.

4. **Block structure for parallel learning:** For faster computing, XGBoost can make use of multiple cores on the CPU. This is possible because of a block structure in its system design. Data is sorted and stored in in-memory units called blocks. Unlike other algorithms, this enables the data layout to be reused by subsequent iterations, instead of computing it again. This feature also serves useful for steps like split finding and column sub-sampling.

5. **Cache awareness:** In XGBoost, non-continuous memory access is required to get the gradient statistics by row index. Hence, XGBoost has been designed to make optimal use of hardware. This is done by allocating internal buffers in each thread, where the gradient statistics can be stored.

6. **Out-of-core computing:** This feature optimizes the available disk space and maximizes its usage when handling huge datasets that do not fit into memory.

#### Algorithm-

1. First, we make a initial prediction like we did in gradient boosting step 1. But by default people take it as 0.5 for both classification and regression. The thick line below represents the 0.5 prediction line.

![image](https://user-images.githubusercontent.com/65160713/131229865-ce55dc11-7a32-4070-b6ff-84020d79a983.png)


Unlike unextreme Gradient Boost which typically uses regular off-the-shelf, Regression Trees. XGBoost uses a unique Regression tree that is called an XGBoost Tree.

1. Then we calculate residuals as we calculated in gradient boosting step-2.
2. Then the tree starts by a single leaf. We place all our residuals on that leaf.
3. Now, we calculate similarity score for our residuals by-

For regression-

![image](https://user-images.githubusercontent.com/65160713/131229895-f61b13f2-5976-4238-868d-77f52a1f7b9c.png)

For classification-

![image](https://user-images.githubusercontent.com/65160713/131229913-3045a0ce-39ad-4d5a-99a0-f5279ac53ea5.png)

_where λ is a regularization parameter._

4. Then we consider whether we could do a better job, clustering similar residuals if we split them into 2 groups. Then we calculate the similarity for each groups (leaf and right). Then we quantify how much better the leaves cluster similar residuals than the root by calculating the gain.

      Gain = Left similarity + Right similarity- Root similarity

After this, we could compare the gain with this and gain with other thresholds to find the **biggest one for better split.**

5. Using the tree with the highest gain, each node will split into further sub-nodes. The nodes will stop splitting when it has only 1 residual left or based on the user defined min number of sample data in each node, max iterations or tree depth.
6. Tree pruning prevents overfitting by comparing the gain value with the complexity parameter γ (for eg γ = 130). A branch containing the terminal node is prune when gain < γ (or gain-γ = negative). Pruning will start from the lowest branch. If gain > γ, the branch and branches before it will not be pruned. If the gain < γ, the lowest level branch is prune and the model will check the pruning conditions on the previous level.
7. Then we calculate the output value of the tree by-

For regression-

![image](https://user-images.githubusercontent.com/65160713/131230069-95844651-e206-441b-97d0-204bfc4dda90.png)

For classification-

![image](https://user-images.githubusercontent.com/65160713/131230076-76fca7cd-c26d-4cb1-9271-4fcf4e7850c0.png)

Output formula is same as similarity score, except we don’t square the residuals.

8. Now, our model is ready and we can make predictions from it. We do it so by-

![image](https://user-images.githubusercontent.com/65160713/131230101-8343a004-6973-4563-869e-d37e013cd4d0.png)

_where ε is the learning rate (default 0.3)._

9. A second tree is built and fitted on the values of Residual 1 and the process will repeat until the residual values become very small, or when a maximum iteration is reached (user defined). The presence of a λ parameter shrinks the similarity score, gain and output value. The smaller values prevents overfitting and increases accuracy and stability (more pruning + smaller contribution from each tree)

It seems recommended to set the tree method of XGBoost to hist when dealing with large datasets, as it decreases the training time a lot without affecting the performance.

For more details refer to xgboost tutorial of statquest with josh stammer.

**_We most of the times use Xgboost as it outperforms other boosting techniques. However, when dataset is very large, we use lightGBM for reducing complexity._**

### CatBoost-

When your categorical variables have too many labels (i.e. they are highly cardinal), performing one-hot-encoding on them exponentially increases the dimensionality and it becomes really difficult to work with the dataset. CatBoost can automatically deal with categorical variables. Unlike most Machine Learning models available today, CatBoost requires minimal data preparation. It also handles missing values, and text variables. It is slower to train but faster to predict than other algorithms. It is comparatively **8x faster** then XGBoost while predicting.

Catboost introduces two critical algorithmic advances in gradient boost — the implementation of ordered boosting, a permutation-driven alternative to the classic algorithm, and an innovative algorithm for processing categorical features. CatBoost can perform very well in situations where the data changes frequently.

Let’s say, we have 10 data points in our dataset and are ordered in time as shown below.

![image](https://user-images.githubusercontent.com/65160713/131230136-a5821384-0e5e-47e6-ba2e-9395537c8b84.png)

If data doesn’t have time, CatBoost randomly creates an artificial time for each datapoint.

**Step 1:** Calculate residuals for each data point using a model that has been trained on all the other data points at that time (For Example, to calculate residual for x5 datapoint, we train one model using x1, x2, x3, and x4 ). Hence we train different models to calculate residuals for different data points. In the end, we are calculating residuals for each data point that the corresponding model has never seen that datapoint before.

**Step 2:** train the model by using the residuals of each data point as class labels

**Step 3:** Repeat Step 1 & Step 2 (_for n iterations_)

For the above toy dataset, we should train 9 different models to get residuals for 9 data points. This is computationally expensive when we have many data points.

Hence by default, instead of training different models for each data point, it trains only log(num_of_datapoints) models. Now if a model has been trained on n data points then that model is used to calculate residuals for the next n data points.

• A model that has been trained on the first data point is used for calculating residuals of the second data point.

• Another model that has been trained on the first two data points is used for calculating residuals of third and fourth data points

• **and so on…**

In the above toy dataset, now we calculate residuals of x5,x6,x7 and x8 using a model that has been trained on x1, x2,x3, and x4.

All this procedure that I have explained until now is known as **ordered boosting**.

CatBoost has a very good vector representation of categorical data. It takes concepts of ordered boosting and applies the same to **response coding**.

In response coding, we represent each categorical feature using the mean to the target values of all the data points with the same categorical feature. We are representing a feature value of the data point with its class label. This leads to target leakage.

CatBoost considers only the previous data points to that time and calculates the mean to the target values of those data points having the same categorical feature. Below is a detailed explanation with examples.

CatBoost vectorize all the categorical features without any target leakage. Instead of considering all the data points, it will consider only data points that are past in time to a data point.

CatBoost combines multiple categorical features. For the most number of times combining two categorical features makes sense. CatBoost does this for you automatically. CatBoost does feature combinations by building a base tree with the root node consisting only a single feature and for the child nodes, it randomly selects the other best feature and represents it along with the feature in the root node.

#### Points to remember-
1. Catboost’s power lies in its **categorical features preprocessing, prediction time and model analysis.**
2. Catboost’s weaknesses are its training and optimization times.
3. Don’t forget to pass cat_features argument to the classifier object. You aren’t really utilizing the power of Catboost without it.
4. One of the cool things about CatBoost is its stability when changing hyperparameters especially when used with large training sets. It gives optimal result with default parameters thereby saving time on parameter tuning. Though Catboost performs well with default parameters, there are several parameters that drive a significant improvement in results when tuned.
5. Working with small datasets — There are some instances when you have less number of data points and you need minimal Log-loss. In those situations you can set parameters fold_len_multiplier as close as to 1 (must be >1) and approx_on_full_history =True . With these parameters, CatBoost calculates residuals for each data point using a different model.
6. For large datasets, you can train CatBoost on GPUs by setting parameter task_type = GPU. It also supports multi-server distributed GPUs. CatBoost also supports older GPUs that you can train it in Google Colabs.
7. By default, CatBoost has an overfitting detector that it stops training when CV error starts increasing. You can set parameter od_type = Iter to stop training your model after few iterations.
8. Like other algorithms, we can also balance an imbalanced dataset with the class_weight parameter.

## Conclusion-

Adaboost and normal gradient boosting are not used much now. Now we are left with catboost, lightGBM and XGBOOST.

So if you have very large datasets then you should use lightGBM as it is much faster than XGBOOST and only little less accurate than XGBOOST(Sometimes can be more accurate also). Also as lightGBM are much faster to train, you can do more of hyperparameter tuning. But if your dataset have categorical variables then lightGBM becomes slow and XGBOOST can’t even handle categorical variables. So then you use Catboost. Also in real world when you need faster predictions or if you are too lazy to do hyperparameter tuning , Catboost is default choice.

In general, it is important to note that a large amount of approaches I’ve seen involve combining all three boosting algorithms in a model stack (i.e. ensembling). LightGBM, CatBoost, and XGBoost might be thrown together as three base learners and then combined via. a GLM or neural network. This is done to really squeeze out decimal places on the leaderboard and so I doubt there is any theoretical (or practical) justification for it besides competitions.

But remember bagging requires cautious tuning of different hyper-parameters.

## Stacking-

Stacking is an ensemble learning technique that combines multiple classification or regression models via a meta-classifier or a meta-regressor. The base level models are trained based on a complete training set, then the meta-model is trained on the outputs of the base level model as features. Stacking learns to combine the base models using a meta-model whereas bagging and boosting combine weak learners following deterministic algorithms.

The base level often consists of different learning algorithms and therefore stacking ensembles are often heterogeneous.

For example first we train decision tree as a base model and then use the prediction output of the decision tree as feature to a meta model such as KNN or any other model, even another decision tree. We can increase it further to give the output of our meta model to another meta model.

Meta Learner is kind of trying to find the optimal combination of base learners. Let us take an example of classification problem where we are trying to classify 4 classes and as a part of traditional paradigm we are testing various models and we find out that Logistic Regression is making better predictions on class 1 data and SVM is making better on class 2 and class 4 and KNN is doing better on class 3 and class 2.This performance is predicted because in general no model is perfect and has its own advantages and disadvantages .So, If we train a model on the predictions of these model can we get better results?. This is the idea on which this entire concept is built upon. So if we train a Random Forest Classifier on these predictions of LR,SVM,KNN we get better results.

![image](https://user-images.githubusercontent.com/65160713/131230237-744c08a2-8973-4fc6-b56e-437bbc07e82e.png)

So, assume that we want to fit a stacking ensemble composed of L weak learners. Then we have to follow the steps thereafter:

1. Split the training data in two folds
2. Choose L weak learners and fit them to data of the first fold
3. For each of the L weak learners, make predictions for observations in the second fold.
4. Fit the meta-model on the second fold, using predictions made by the weak learners as inputs

In the previous steps, we split the dataset in two folds because predictions on data that have been used for the training of the weak learners are not relevant for the training of the meta-model. Thus, an obvious drawback of this split of our dataset in two parts is that we only have half of the data to train the base models and half of the data to train the meta-model. In order to overcome this limitation, we can however follow some kind of “k-fold cross-training” approach (similar to what is done in k-fold cross-validation) such that all the observations can be used to train the meta-model: for any observation, the prediction of the weak learners are done with instances of these weak learners trained on the k-1 folds that do not contain the considered observation.

Base-models are often complex and diverse. As such, it is often a good idea to use a range of models that make very different assumptions about how to solve the predictive modeling task, such as linear models, decision trees, support vector machines, neural networks, and more. Other ensemble algorithms may also be used as base-models, such as random forests.

The meta-model is often simple, providing a smooth interpretation of the predictions made by the base models. As such, linear models are often used as the meta-model, such as linear regression for regression tasks (predicting a numeric value) and logistic regression for classification tasks (predicting a class label). Although this is common, it is not required.

Stacking is used in competitions and perform really good there. Also, by default bagging and boosting work with decision trees but there’s no such rule in stacking.

**Stacking is appropriate when multiple different machine learning models have skill on a dataset, but have skill in different ways.** Another way to say this is that the predictions made by the models or the errors in predictions made by the models are uncorrelated or have a low correlation.

Usually there is a very less difference between stacking and boosting when used at their best. So there’s no general rule for which to use. You can use any of them as they give nearly same accuracy most of the times. But generally **boosting and stacking outperform bagging.**

For seeing it’s implementation, you can refer to https://machinelearningmastery.com/implementing-stacking-scratch-python/

### Multi-levels Stacking-

A possible extension of stacking is multi-level stacking. It consists in doing **stacking with multiple layers**. As an example, let’s consider a 3-levels stacking. In the first level (layer), we fit the L weak learners that have been chosen. Then, in the second level, instead of fitting a single meta-model on the weak models predictions (as it was described in the previous subsection) we fit M such meta-models. Finally, in the third level we fit a last meta-model that takes as inputs the predictions returned by the M meta-models of the previous level.

From a practical point of view, notice that for each meta-model of the different levels of a multi-levels stacking ensemble model, we have to choose a learning algorithm that can be almost whatever we want (even algorithms already used at lower levels). We can also mention that adding levels can either be data expensive (if k-folds like technique is not used and, then, more data are needed) or time expensive (if k-folds like technique is used and, then, lot of models need to be fitted).

### Blending-

Blending follows the same approach as stacking but uses only a holdout (validation) set from the train set to make predictions. In other words, unlike stacking (which did prediction on both training and test set), the predictions are made on the holdout set only. The holdout set and the predictions are used to build a model which is run on the test set.

Very roughly, we can say that bagging will mainly focus at getting an ensemble model with less variance than its components whereas boosting and stacking will mainly try to produce strong models less biased than their components (even if variance can also be reduced).

## Practical training-

**Dataset**= data_banknote_authentication

**Size**=  (1372,4)

**Task**= Binary classification

**Class ratio** =[762, 610]
    xtrain,xtest,ytrain,ytest=train_test_split(X,target,test_size=0.4,random_state=42)

    xtest,xval,ytest,yval=train_test_split(xtest,ytest,test_size=0.5,random_state=42)

Preprocessing= Scaling by MinMaxScaler

### Decision Tree=

random_state=42

Without any hyperparameter tuning, accuracies are —

    Test accuracy is  98.17518248175182 %
    
    Train accuracy is  100.0 %
    
    Validation accuracy is  97.45454545454545 %

Now, first of all as we know we have class imbalance problem, let’s take advantage of class_weight parameter-

#### class_weight = ‘balanced’
    
    Test accuracy is  98.54014598540147 %
    
    Train accuracy is  100.0 %
    
    Validation accuracy is  98.18181818181819 %

As we know class 1 has1.25 times more sample than class 0, so we can also use **class_weight = {0:1,1:1.25}**

    Test accuracy is  98.54014598540147 %
    
    Train accuracy is  100.0 %
    
    Validation accuracy is  98.54545454545455 %

Using a class weighting that is the inverse ratio of the training data is just a heuristic. It is possible that better performance can be achieved with a different class weighting, and this too will depend on the choice of performance metric used to evaluate the model.

But as we keep on increasing the class ratio of a particular class keeping the other one constant, the decision tree starts to skew in the class’s side. And when we use inverse class ratio as we used in the above one, we get the centered tree.

**class_weight=** {0:1,1:2}
    
    Test accuracy is  99.63503649635037 %
    
    Train accuracy is  100.0 %
    
    Validation accuracy is  98.9090909090909 %

**class_weight=** {0:1,1:60}
    
    Test accuracy is  98.90510948905109 %
    
    Train accuracy is  100.0 %
    
    Validation accuracy is  99.63636363636364 %

**{0:2,1:1}**-
    
    Test accuracy is  99.63503649635037 %
    
    Train accuracy is  100.0 %
    
    Validation accuracy is  98.9090909090909 %

But let’s continue on without using this parameter and see how our model perfoms-

Without any parameter-

    Test accuracy is 98.18181818181819 %
    
    Train accuracy is 100.0 %

If using **criterio=’entropy’** -
    Test accuracy is 98.54545454545455 %
    
    Train accuracy is 100.0 %

So, we will use entropy metric onwards.

**_If you think that your tree is overfitting then it is better to first tune min_samples_split or min_samples_leaf than max_depth. As, most of the times, these parameter help to reduce training accuracy without much hurting test accuracy. While when we use max_depth parameter, it reduces both.
_**

#### Q. What’s difference between _min_samples_leaf_ and _min_samples_split_?

**Ans.** _min_samples_split_ specifies the minimum number of samples required to split an internal node, while _min_samples_leaf_ specifies the minimum number of samples required to be at a leaf node. For instance, if _min_samples_split_ = 5, and there are 7 samples at an internal node, then the split is allowed. But let’s say the split results in two leaves, one with 1 sample, and another with 6 samples. If _min_samples_leaf_ = 2, then the split won’t be allowed (even if the internal node has 7 samples) because one of the leaves resulted will have less then the minimum number of samples required to be at a leaf node. Mostly value of _min_samples_split_ is greater than _min_samples_leaf_.

#### ccp_alpa-

**ccp_alpha**= pruning parameter, default=0

If we increase it’s value, both training and test accuracy starts to decrease. It means we don’t require pruning

Other parameters don’t help much in improving accuracy so we will not discuss them.

### Random Forest=

**random_state**= 42

Accuracies by default=

    Test accuracy is  98.90510948905109 %
    
    Train accuracy is  100.0 %
    
    Validation accuracy is  99.27272727272727 %

#### n_estimators=

![image](https://user-images.githubusercontent.com/65160713/131230320-6604339b-9ed5-4fe9-b99d-e31f0c0f6f9d.png)

We get our accuracy as-

    Test accuracy is  98.90510948905109 %
      
    Train accuracy is  100.0 %
      
    Validation accuracy is  99.27272727272727 %

In our dataset there is a class imbalance problem so let’s use **class_weight**= ’balanced’-

![image](https://user-images.githubusercontent.com/65160713/131230324-78c033d3-5ce5-498f-94dd-3777035b40ce.png)

So, we get max accuracy at **n_estimators**= 7-

    Test accuracy is  100.0 %
      
    Train accuracy is  100.0 %
      
    Validation accuracy is  99.63636363636364 %

Let’s try **criterion**= ”entropy”-

    model=RandomForestClassifier(n_estimators=5,criterion=’entropy’,random_state=42,class_weight=’balanced’)-
        
    Test accuracy is  99.63503649635037 %
      
    Train accuracy is  100.0 %
      
    Validation accuracy is  99.63636363636364 %

As our accuracy decreases, so we continue to use gini for this dataset.

Now let’s tune **min_samples_split**-

When we use _min_samples_split_= 7, we achieve the milestone- _n_estimators_= 5, _criterion_= ’gini’, _random_state_= 42, _class_weight_= ’balanced_subsample’, _min_samples_split_=7

    Test accuracy is  100.0 %
    
    Train accuracy is  100.0 %
      
    Validation accuracy is  100.0 

We generally don’t use large values of _min_samples_split_ and _min_samples_leaf_.

**Now let’s see the affect of other parameters-**

#### max_samples-

It tells us maximum samples to consider while building decision trees. Higher it’s value, more stable the accuracy curve is as less is it’s randomness.

#### min_samples_leaf-

It has to be tried on different values like _min_samples_split_ parameter.

#### ccp_alpha-

Used to prune tree. Used only when tree is overfitting. It can give you better result than tuning parameters like _max_depth_, _max_samples_, _min_samples_split_, _min_samples_leaf_.

#### max_depth-

It should be tried after trying above 3 parameters because it solves the problem of overfitting but also decreases testing accuracy.

Other parameters don’t give much improvement in accuracy, so we will not try that.
