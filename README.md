

# Relating Social Media to Stock Movements_DA-31st-December
Both social media sentiment and stock market data are crucial for stock price prediction. 
So, in this project we analyzed the dynamics of stock markets based on both social media news (text data) and stock prices (numerical data).


## Understanding the Dataset
The dataset we are working on is a combination of **Wallstreetbets-Reddit
news** and **the Standard & Poor’s 500 (S&p 500) stock
price** from **2013** to **2018**.

- The news dataset contains the top
**25** news from **Reddit** on each day from **2013** to **2018**. 

- The **S&P 500** contains the core stock market information for each day
such as **Open**, **Close**, and **Volume**. 

- The SCORE of the dataset is whether the stock price is **increase** (labeled as **1**) or **decrease**
(labeled as **0**) on that day.


## EDA

**Introduction:**

- **data** dataset comprises 5698
 rows and 8 columns.
- Dataset consists of continuous variable and float data type. 
- Dataset column variables 'Open', 'Close', 'High', 'Low', 'Volume', are the stock variables from historical dataset and other variables are showing polarity  of news which are the derived variables using sentiment analysis as discussed in the above section.










**Descriptive Statistics:**

Using **describe()** we could get the following result for the numerical features


open	high	low	close	volume	
count	5697.000000	5697.000000	5697.000000	5698.000000	5.698000e+03
mean	88.139399	89.012936	87.245609	88.146015	1.718703e+06
std	32.666995	32.960833	32.363413	32.660301	1.248357e+06
min	30.380000	31.090000	29.730000	29.940000	1.000000e+02
25%	64.650000	65.310000	64.053300	64.672500	9.880475e+05
50%	80.750000	81.490000	79.990000	80.750000	1.460298e+06
75%	105.270000	106.270000	104.350000	105.345000	2.135991e+06
max	201.240000	201.240000	198.160000	200.380000	3.378024e+07


## Preprocessing and Sentiment Analysis

We filled out the NaN values in the missed three topics. And got the polarity and subjectivity for the news' topics. **Polarity** is of **'float'** type and lies in the range of **-1**, **1**, where **1** 



means a **high positive** sentiment, and **-1** means a **high negative** sentiment.

So, they will be very helpful in determining the increase or decrease of the stock market.

Then we checked the missing values in the stock market information, it was complete.
Then we merged the sentiment information (**polarity** ) by **date** with the stock market information (**Open**, **High**, **Low**, **Close**, **Volume**, **Adj Close**) in **merged_data** dataframe.

Before modeling and after splitting we scaled the data using standardization to shift the distribution to have a mean of zero and a standard deviation of one.
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
rescaledValidationX = scaler.transform(X_valid)
```
**fit_transform()** is used on the training data so that we can scale the training data and also learn the scaling parameters of that data. Here, the model built by us will learn the mean and variance of the features of the training set. These learned parameters are then used to scale our test data.

**transform()** uses the same mean and variance as it is calculated from our training data to transform our test data. Thus, the parameters learned by our model using the training data will help us to transform our test data. As we do not want to be biased with our model, but we want our test data to be completely new and a surprise set for our model.

## Preprocessing Again

Now, after observing the outliers in **polarity** of a lot of topics, we decided to concatenate all the 14 topics in one paragraph,
then we can get only one column for **polarity**. So, we merged these data again with the stock market numerical information and got **merged_data** 
dataframe, then scaled it.



## Model Building

#### Metrics considered for Model Evaluation
**Accuracy , Precision , Recall and F1 Score**
- Accuracy: What proportion of actual positives and negatives is correctly classified?
- Precision: What proportion of predicted positives are truly positive ?
- Recall: What proportion of actual positives is correctly classified ?
- F1 Score : Harmonic mean of Precision and Recall

#### Logistic Regression
- Logistic Regression helps find how probabilities are changed with actions.
- The function is defined as P(y) = 1 / 1+e^-(A+Bx) 
- Logistic regression involves finding the **best fit S-curve** where A is the intercept and B is the regression coefficient. The output of logistic regression is a probability score.


### Choosing the features
After choosing model based on confusion matrix here where **choose the features** taking in consideration the deployment phase.

We know from the EDA that all the features are highly correlated and almost follow the same trend among the time.
So, along with polarity and subjectivity we choose the open price with the assumption that the user knows the open price but not the close price and wants to figure out if the stock price will increase or decrease.

When we apply the **logistic regression** model accuracy dropped from 80% to 55%.
So, we will use both **Open** and **Close** and exclude **High,	Low, Volume, Adj Close**.
```merged_data2 = merged_data2[['Label', 'polarity', 'subjectivity', 'Open', 'Close']]
precision    recall  f1-score   support

           0       1.00      1.00      1.00   2563950
           1       0.00      0.00      0.00       968

    accuracy                           1.00   2564918
   macro avg       0.50      0.50      0.50   2564918
weighted avg       1.00      1.00      1.00   2564918







