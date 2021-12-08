 <p align="center"> Team Members: <br/> Maxwell Chen, Brian Epstein, Rachel Kaplan, Kaitlin Evans </p>

### Links:
<a href="https://github.com/MaxwellXChen/Machine-Learning-and-Movies"> View on Github </a> <br/>
[Project Proposal](proposal.md) <br/>
[Midterm Report](midtermReport.md)   

## Final Report

### Introduction/Background
The movie industry is an essential part of recreation and entertainment for billions of people. Over the past couple of decades, however, the industry has completely changed. The internet has introduced new ways for movies and shows to be reviewed and rated. Now, any person has the ability to rate and review a film or show they have watched and share that rating with others who may be interested. Entertainment database IMDb has become “the most popular website for movie ratings and movie reviews.”(Kharb, 2020) These ratings, along with ones from professional entertainment critics, can predict how well a production will perform and therefore the profit that it generates. While ratings are an essential part of a movie’s performance, the producer’s main point of interest is the gross income, “predicted revenues can be used for planning both the production and distribution stages.” (Dey, 2016) Right before, and for a period of time after a movie’s launch, marketing becomes the most important tool to ensure a movie gets the exposure necessary to be successful. In order to gauge a movie’s projected success rate based on the initial reviews and ratings, and not overspend on marketing, producers can use our proposed prediction model.

### Problem Definition
Given various parameters of a movie, such as genre, Metascore, IMDb score, budget, runtime, and production country, our project aims to predict the gross income of the film. This serves to provide movie producers with the information that will be useful in predicting the critical and commercial success of projects that are in development, allowing them to optimize their movie’s parameters to maximize profits. 

Additionally, the models will be applied to the task of predicting the IMDb score of a movie. This would be useful in determining a movie's potential popularity among internet reviewers, which could be an early indicator of a movie's success or failure at the box office.

### Data Collection
The data used for our project was collected using the kaggle dataset "IMDb movies extensive dataset". This dataset contains 85,855 movies and 22 unique attributes. The movies chosen had received more than 100 votes towards their average IMDb score as of 01/01/2020.

### Methods

#### Data cleanup
The original data was sparse, and had around 85,000 entries. In order to use this dataset, sparse entries first needed to be removed. Initially we identified features in the data that could not easily be converted to a numerical value, such as the movie description, writer, and actors, and removed these features. Next we went through and removed entries that did not have data for each of the remaining features. After this step, the char vectors that represented numbers like budget and gross income needed to be converted to double values. Finally, features like language and country were converted from lists of the actual languages and countries to the quantity of languages and countries for each entry. This conversion was made on the presumption that a movie that was released in more languages or countries would appeal to a wider audience, thus making these features relevant to our model.

#### Data preprocessing 
Data preprocessing for the IMDb score predictions was limited, as the input data already had few features. This was because the nature of attempting to predict the score based only on information known at the release of the movie removes some features such as number of votes and gross income, inherently.

For Data preprocessing, we chose to employ 4 different methods using our cleaned dataset. We implemented forward feature selection and backward feature selection as well as the dimensionality reduction methods PCA and lasso regression. When implementing these methods, we dropped the features title, original title, and usa gross income as they disrupted the results of our methods. 

##### Forward and Backward Feature Selection
We chose to run forward and backward feature selection to assist us in determining the significant features that should be used in our model. Through our research, we determined that feature selection would be useful for our project. (Kim et al., 2020) Forward and backward feature selection allow us to find the features with a p-value less than our significance level of 0.05 (95% confidence). With this method, our goal is to see which features are important to implement for success with our prediction model. 

##### PCA
We chose to use PCA to enhance variation and identify strong patterns in our dataset. We wanted to identify the first two principal components to get a better visual depiction of our data and see if there were any strong patterns identified by the first two principal components. We also wanted to use this type of dimensionality reduction because we had 15 features to begin with and wanted to identify how many features were needed to create a strong model. After running PCA, we chose to identify the explained variance ratio to understand if the two principal components would be enough to explain the variation in the model, or if we would need to have a higher number of principal components. Our goal was to identify the number of principal components that would explain 95% of the variation in the data.

##### LASSO Regression
LASSO provides a simple way to reduce the number of features in a model. Using the Lasso method allows the inclusion of a penalty factor, usually selected through cross validation,which is used during the penalization of the L1 norm of the weights. Compared to stepwise feature selection, the LASSO method’s inclusion of cross validation helps assure that the generated model will respond well to more generalized future input. It is also useful in that it provides a concrete feature importance coefficient for each feature. These coefficients are useful in informing the decision about which features to include in the model.

#### Models
For our machine learning model, we chose to employ 3 different methods of supervised learning. We implemented principal component regression and random forest regression as well as a neural network. 

##### Principal Component Regression
After running PCA, we chose to run linear regression to estimate the unknown regression coefficients of our data in a linear regression model using the principal components.We will find whichever number of principal components retain 95% of the variance, and then we will use those principal components to run linear regression on the dataset. Our goal is to create a model using PCA and PCR that will predict the income of a movie based on its features with high accuracy.

##### Random Forest Regression
Another attempted method was Random Forest Regression. Testing and training sets will be created using an 80% training, 20% testing split. 

##### Neural Network
We chose to build a neural network model as we found through our research that “the model built by neural networks [does] a better job of predicting box office [performance]” (Hsu et al., 2014)

### Results and Discussion

#### Data cleanup
The data clean up resulted in a dataset containing 6083 movies with 15 features. The features are title, original title, year, genre, duration, country of origin, language, IMDb score (avg_vote), Metascore, number of votes, budget, USA gross income, worldwide gross income, number of user reviews, and number of critic reviews.

![unnamed](https://user-images.githubusercontent.com/40035500/145119967-c6adb971-ba98-4e88-a8ad-5530caed026e.png)

#### Data preprocessing

##### Forward and Backward Feature Selection
Forward and backward feature selection were run on IMDb score. The target returned the following features.

<img width="752" alt="imdb" src="https://user-images.githubusercontent.com/40035500/145153687-6a3afe16-e0ff-49c1-94cb-ed3038627d72.png">

Looking at these results, it is interesting that the IMDb score returned year and duration as some of the most important features for both forward and backward feature selection. This shows that movie length has a large impact on score and people are probably more critical of very long or very short movies. Additionally, older movies are probably rated very differently from newer movies as they were originally rated and reviewed in a very different setting than how newer movies are rated and reviewed. 


Forward and backward feature selection were run on worldwide gross income. The target returned the following features. 

<img width="770" alt="Screen Shot 2021-12-07 at 6 00 38 PM" src="https://user-images.githubusercontent.com/40035500/145120049-b64b9a67-b83e-4f3d-b32d-fe07bc0e4a55.png">

Looking at these results, it makes sense that reviews -both IMDb and metascore- should be important features when predicting the model as they can provide a general idea of the popularity of a movie. However, it is interesting that a movie’s budget would have an impact on the success of a movie. Perhaps the budget would assume more marketing materials and therefore more exposure. 

##### PCA
PCA was performed on the dataset to determine patterns in the data for the IMDb scores. First, we calculated the first and second principal components to visualize the data and identify any strong patterns. Reducing the number of features from 15 to 2 was helpful in visualizing our data, but the results determined that 2 features was not enough to retain a large portion of the variance in the data. Calculating the first and second principal components resulted in a very low explained variance, which indicated that only a small amount of variance was explained by the first two principal components.

![PCA_Visualization_IMDB](https://user-images.githubusercontent.com/72058559/145155346-ca80c5e1-0f33-43f3-9e96-f6ff69429957.PNG)
![ExpectedVar_2PCs_IMDB](https://user-images.githubusercontent.com/72058559/145155365-3a464b23-3eda-4317-bb2e-a82fdd22808f.PNG)

The explained variance of the first two principal components was [0.29740404, 0.14722416].
The explained variance of the first principal component is about 30%, and the second is close to 15%, indicating that just the first two principal components do not retain much of the variance. While it was helpful to visualize our data, we decided that reducing the data to only the first two principal components would not be a useful model.

This indicated that a larger number of principal components were required for our data. We then chose to run PCA again, this time with the goal of explaining 95% of the variance. 

![ExpectedVar0 95_IMDB](https://user-images.githubusercontent.com/72058559/145155414-216c8d7e-cc3c-413b-bf1c-750a17bab5f5.PNG)

After performing PCA, the results determined that the first 7 principal components retained 95% of the variance in the data for the IMDb score predictive model.

Next, we followed the same process for the Worldwide Gross Income data predictions. We first visualized the first two principal components to see if there were any strong trends.

![PCA_Visualization_Income](https://user-images.githubusercontent.com/72058559/145155471-6e69553b-bda5-49d2-abd8-73db939802e5.PNG)
![ExpectedVar_2PCs_Income](https://user-images.githubusercontent.com/72058559/145155617-5ee2fed4-7d19-4e4d-b7f0-03d481f8e3ca.PNG)

The explained variance of the first two principal components was [0.30134129, 0.14838675]. Similar to the results for the IMDb score, the explained variance for the first and second principal components were roughly 30% and 15%, respectively. We chose to run PCA again, this time with the goal of explaining 95% of the variance. 

![ExpectedVar0 95_Income](https://user-images.githubusercontent.com/72058559/145155635-612dd4b4-7475-43b1-aab2-0681184126f3.PNG)

After performing PCA, the results determined that the first 9 principal components retained 95% of the variance in the data for the IMDb score predictive model.

##### LASSO Regression

LASSO selection was run on the preprocessed data with the goal of identifying important features to include in the data we run through our model. The input data for the LASSO selection model was first normalized, as were the output values for gross income. As shown in the figure below, budget and number of votes were the two most important features as selected by SFM with an importance threshold specifying a minimum coefficient of .2. This makes sense rationally as it follows that a movie that grosses well also generates a lot of interest, which leads to votes on review platforms. It also follows that movies with larger budgets usually gross more money. When using sequential feature selection with lasso regression with 3 features to select, votes, budget, and metascore were selected for both forward and backward selection.
![lasso](https://user-images.githubusercontent.com/40035500/145120494-04847613-d25d-4436-b9d2-22735d7cd0cc.png)

#### Models

##### Principal Component Regression
For our IMDb score predictive model, the results of PCA were that 95% of the variance was retained in the first 7 principal components. Using these 7 components, we chose to run regression to predict the IMDb score of a movie. Once we had a regression model, we were able to use the model to predict the IMDb score for our test data and compare it with the actual values. 

Note that the below graph utilizes the first principal component to plot the data, so it is not a complete visualization since we cannot visualize 7 dimensions. However, below is a visualization of the first principal component with the actual and predicted gross income of the movie.

![PCRModel_IMDB](https://user-images.githubusercontent.com/72058559/145155732-4950fb7e-7ef6-4927-81e2-5d4dcf3f54f7.PNG)

As seen, the model did relatively well in portraying the shape of the data. Additionally, the explained variance score for this model was 0.26952, which means that roughly 27% of the variation in the dataset is explained by this model. Additionally, the mean squared error value for this data was 0.72776, which is relatively low and indicates a pretty good model.

For the worldwide gross income predictive model, the results of PCA were that 95% of the variance was retained in the first 9 principal components. Using these 9 principal components, we chose to run regression to predict the worldwide gross income of a movie. We used this model to predict the gross income for our test data and compare it with the actual values. Again, the graph below is just the first principal component plotted versus the worldwide gross income.

![PCRModel_Income](https://user-images.githubusercontent.com/72058559/145155769-945beafe-a5c7-4f0c-abb9-0a2a545cc564.PNG)

As seen, the model did relatively well in portraying the shape of the data. Additionally, the explained variance score for this model was 0.6593772, which means that roughly 66% of the variation in the dataset is explained by this model. 

##### Random Forest Regression
We performed a random forest regression model to determine if this would be a better fit for our data. After running this model, we looked at the importance of all the features to determine which would be the best feature to predict the income. 

![rfr](https://user-images.githubusercontent.com/40035500/145120743-80626990-6920-4414-a224-48a3fc055bd3.png)

We found that budget was actually the feature with the highest importance (0.52). A graph was then made of the predicted and actual worldwide gross income values for each of the movies in the testing dataset against the budget of the movies, which was the feature with the highest importance.

![rfr2](https://user-images.githubusercontent.com/40035500/145120773-c92e8b93-ec0f-4f65-a836-67b4440943e4.png)

The explained variance score for this model was 0.730152, which means that roughly 73% of the variance in the dataset is explained by the model. Compared to the Principal Component Regression method, Random Forest Regression explains a slightly larger percentage of the variance, but both could be better.

We also performed Random Forest Regression to predict the IMDb score.

<img width="403" alt="imdb score" src="https://user-images.githubusercontent.com/40035500/145153741-d8956c03-5131-4d29-a335-1063803eff9f.png">

We found that the model did very well, while it only had the 40% for explained variance, the accuracy was nearly 90%, with the both mean squared and mean absolute error very low. Overall, the model performed very well with IMDb score, even though it did not have as high of an explained variance as worldwide gross income. 

##### Neural Network

Model setup: Before even thinking about building the model, we normalized all data to a 0 to 1 range using sklearn.preprocessing’s MinMaxScaler library. Next, we would “bundle” up all the features we were considering: Number of genres, number of countries, runtimes, and budgets. In addition, we determined it appropriate to split the training and testing/validation set into an 80-20 split. 

Our scaling and training/validation splitting process:

![Normalize and Splitting](https://user-images.githubusercontent.com/41342635/145160411-bb17587b-57ec-45ef-b1ab-ce22990fde37.png)

Model Building: The input to our model would be a list of these bundled up features as a singular input, and the output being an IMDB score but scaled from 0 to 1. We built our sequential model from Keras and began with a Dense layer with the “relu” activation function, which we felt was standard and a safe bet for every model. We then have two larger Dense layers with sigmoid activation functions as we are ultimately attempting to predict a score from 0 to 1 (IMDB scaled down by a factor of 10) and sigmoid does that perfectly as, intuitively, the graph itself falls in the range 0 to 1 due to the horizontal asymptotes. In our model.compile function call, we found that mean_squared_logarithmic_error gave us the best results, and we used an adam optimizer (which is also quite standard). Finally, we ran 100 epochs each with a batch size of 50.

Our Findal Model:

![Sequential model](https://user-images.githubusercontent.com/41342635/145160633-63b58cf6-363c-4177-97c0-4a517a6ae193.png)

Our Model Training Process:

![image](https://user-images.githubusercontent.com/41342635/145161019-69314e0a-2979-4ada-9029-28259ad6634c.png)

The loss we achieved with the model ended up being a solid 0.0035, which led to an effective model for predicting IMDB scores.


Accuracy calculation: In determining the accuracy of our model, we used quite a simple method. A predicted output would be classified as accurate if the positive difference between it and the actual value falls within a specified threshold. For example, if our error threshold is 0.1, meaning an IMDB score difference of 1, a predicted point would count as accurate if abs(predicted-actual) <= 0.1.

Our accuracy method and calculation:

![Accuracy](https://user-images.githubusercontent.com/41342635/145160780-f39bb169-73ea-4e28-980f-e6ecd978f04f.png)


Prediction results: With an error threshold of 0.1, our accuracy function deemed ~77% of predicted values as accurate. With an error threshold of 0.05, our accuracy function deemed ~50% of the predicted output as accurate. In addition, removing budgets as a parameter actually raised the accuracy to 78%, which is likely explained by the high variance within budgets, causing the normalization process to be quite awkward as some values could go as low as 10e-7.

Note: For all code on the neural network, please navigate to the “Maxwell” branch and open “MLP.ipynb”.


### Conclusion

When comparing our models, we see some major similarities in that they all perform pretty well when predicting IMDB scores (with Random Forest regression being the best performer), and perform poorly when it comes to predicting worldwide gross income. We believe this is due to the fact that IMDB scores are much easier to normalize because a range of 1-10 can easily be normalized to 0-1, whereas a normalization process with worldwide gross income gets complicated, as we see in our dataset that the range of this feature goes from as low as $95 to as high as $2.8 billion. Therefore, fitting that range into a tiny confinement of 0-1 would be a nightmare, as we would have extremely small numbers that eventually only serve to skew the model’s predictive capabilities. All in all, our team found through our models that, based on our feature selection criteria, it’s extremely easy to find similarly-classified movies with close to identical features. The model would then, of course, treat those two inputs as essentially identical. However, two similarly-classified movies could have wildly different worldwide gross incomes due to the nature of the movie industry with how big-budget blockbuster productions usually perform well in worldwide theaters, whereas small indie productions wouldn’t have as expansive of a reach. In addition, the quality of the movie is not usually indicative of its box office performance, and many similarly-classified movies could perform very differently based on how much is invested in post-production marketing. Therefore, there’s a good amount of factors that we weren’t able to include in our models due to limitations with the dataset, but the models presented above are, in our opinion, an effective yet intuitive way to gauge the potential success of a movie.


