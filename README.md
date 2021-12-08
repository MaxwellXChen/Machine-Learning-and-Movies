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

### Data Collection
The data used for our project was collected using the kaggle dataset "IMDb movies extensive dataset". This dataset contains 85,855 movies and 22 unique attributes. The movies chosen had received more than 100 votes towards their average IMDb score as of 01/01/2020.

### Methods

#### Data cleanup
The original data was sparse, and had around 85,000 entries. In order to use this dataset, sparse entries first needed to be removed. Initially we identified features in the data that could not easily be converted to a numerical value, such as the movie description, writer, and actors, and removed these features. Next we went through and removed entries that did not have data for each of the remaining features. After this step, the char vectors that represented numbers like budget and gross income needed to be converted to double values. Finally, features like language and country were converted from lists of the actual languages and countries to the quantity of languages and countries for each entry. This conversion was made on the presumption that a movie that was released in more languages or countries would appeal to a wider audience, thus making these features relevant to our model.

#### Data preprocessing 
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
PCA was performed on the dataset to determine patterns in the data. First, we calculated the first and second principal components to visualize the data and identify any strong patterns. Reducing the number of features from 15 to 2 was helpful in visualizing our data, but the results determined that 2 features was not enough to retain a large portion of the variance in the data. Calculating the first and second principal components resulted in a very low explained variance, which indicated that only a small amount of variance was explained by the first two principal components.

![pca](https://user-images.githubusercontent.com/40035500/145120268-74de140c-af33-4a19-9d12-baa7abf51a2a.png)
![pca 2](https://user-images.githubusercontent.com/40035500/145120331-52a4e501-16cb-4f91-89cd-1581f374717b.png)

The explained variance of the first two principal components was [0.30134129, 0.14838675].
The explained variance of the first principal component is about 30%, and the second is close to 15%, indicating that just the first two principal components do not retain much of the variance. While it was helpful to visualize our data, we decided that reducing the data to only the first two principal components would not be a useful model.

This indicated that a larger number of principal components were required for our data. We then chose to run PCA again, this time with the goal of explaining 95% of the variance. 

![pca3](https://user-images.githubusercontent.com/40035500/145120392-17caadf8-f2b9-4aee-8082-25347f52631a.png)

After performing PCA, the results determined that the first 9 principal components retained 95% of the variance in the data. In order to retain 95% of the variance, we need 9 features, which are aligned with our results from forward and backward feature selection.

##### LASSO Regression

LASSO selection was run on the preprocessed data with the goal of identifying important features to include in the data we run through our model. The input data for the LASSO selection model was first normalized, as were the output values for gross income. As shown in the figure below, budget and number of votes were the two most important features as selected by SFM with an importance threshold specifying a minimum coefficient of .2. This makes sense rationally as it follows that a movie that grosses well also generates a lot of interest, which leads to votes on review platforms. It also follows that movies with larger budgets usually gross more money. When using sequential feature selection with lasso regression with 3 features to select, votes, budget, and metascore were selected for both forward and backward selection.
![lasso](https://user-images.githubusercontent.com/40035500/145120494-04847613-d25d-4436-b9d2-22735d7cd0cc.png)

#### Models

##### Principal Component Regression
The results of PCA were that 95% of the variance was retained in the first 9 principal components. Using these 9 principal components, we chose to run regression to predict the worldwide gross income of a movie. Once we had a regression model, we were able to use the model to predict the gross income for our test data and compare it with the actual values. Note that the below graph utilizes the first principal component to plot the data, so it is not a complete visualization since we cannot visualize 9 dimensions. However, below is a visualization of the first principal component with the actual and predicted gross income of the movie.

![pcr](https://user-images.githubusercontent.com/40035500/145120654-bc88f3ab-81a0-4e5a-8ab8-243e530ceaa7.png)

As seen, the model did relatively well in portraying the shape of the data. Additionally, the explained variance score for this model was 0.6593771996946548, which means that roughly 66% of the variation in the dataset is explained by this model. 

##### Random Forest Regression
We performed a random forest regression model to determine if this would be a better fit for our data. After running this model, we looked at the importance of all the features to determine which would be the best feature to predict the income. 

![rfr](https://user-images.githubusercontent.com/40035500/145120743-80626990-6920-4414-a224-48a3fc055bd3.png)

We found that budget was actually the feature with the highest importance (0.52). A graph was then made of the predicted and actual worldwide gross income values for each of the movies in the testing dataset against the budget of the movies, which was the feature with the highest importance.

![rfr2](https://user-images.githubusercontent.com/40035500/145120773-c92e8b93-ec0f-4f65-a836-67b4440943e4.png)

The explained variance score for this model was 0.7301520036631554, which means that roughly 73% of the variance in the dataset is explained by the model. Compared to the Principal Component Regression method, Random Forest Regression explains a slightly larger percentage of the variance, but both could be better.

We also performed Random Forest Regression on IMDb score to see how our original problem statement fit into the new model. 

<img width="403" alt="imdb score" src="https://user-images.githubusercontent.com/40035500/145153741-d8956c03-5131-4d29-a335-1063803eff9f.png">

We found that the model did very well, while it only had the 40% for explained variance, the accuracy was nearly 90%, with the both mean squared and mean absolute error very low. Overall, the model performed very well with IMDb score, even though it did not have as high of an explained variance as worldwide gross income. 


##### Neural Network

### Conclusion



