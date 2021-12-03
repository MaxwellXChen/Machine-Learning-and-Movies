 <p align="center"> Team Members: <br/> Maxwell Chen, Brian Epstein, Rachel Kaplan, Kaitlin Evans </p>

### Links:
<a href="https://github.com/MaxwellXChen/Machine-Learning-and-Movies"> View on Github </a> <br/>
[Project Proposal](proposal.md) <br/>
[Midterm Report](#midterm-report)   

## Midterm Report

### Introduction/Background
The movie industry is an essential part of recreation and entertainment for billions of people. Over the past couple of decades, however, the industry has completely changed. The internet has introduced new ways for movies and shows to be reviewed and rated. Now, any person has the ability to rate and review a film or show they have watched and share that rating with others who may be interested. Entertainment database IMDb has become “the most popular website for movie ratings and movie reviews.”(Kharb, 2020) These ratings, along with ones from professional entertainment critics, can decide how well a production performs and therefore the profit that it generates. 

### Problem Definition 
Given various parameters of a movie, such as genre, budget, runtime, and production country, our project aims to predict the Metascore and IMDb rating of the film. This serves to provide movie producers with the information that will be useful in predicting the critical and commercial success of projects that are in development, allowing them to optimize their movie’s parameters to maximize ratings. 

### Data Collection 
The data used for our project was collected using the kaggle dataset "IMDb movies extensive dataset". This dataset contains 85,855 movies and 22 unique attributes. The movies chosen had received more than 100 votes towards their average IMDb score as of 01/01/2020.

### Methods
#### Data Cleanup
The original data was sparse, and had around 85,000 entries. In order to use this dataset, sparse entries first needed to be removed. Initially we identified features in the data that could not easily be converted to a numerical value, such as the movie description, writer, and actors, and removed these features. Next we went through and removed entries that did not have data for each of the remaining features. After this step, the char vectors that represented numbers like budget and gross income needed to be converted to double values. Finally, features like language and country were converted from lists of the actual languages and countries to the quantity of languages and countries for each entry. This conversion was made on the presumption that a movie that was released in more languages or countries would appeal to a wider audience, thus making these features relevant to our model.

#### Data Preprocessing
For Data preprocessing, we chose to employ 4 different methods using our cleaned dataset. We implemented forward feature selection and backward feature selection as well as the dimensionality reduction methods PCA and lasso regression. When implementing these methods, we dropped the features title, original title, and year as they disrupted the results of our methods. 

##### Forward and Backward Feature Selection
We chose to run forward and backward feature selection to assist us in determining the significant features that should be used in our model. Through our research, we determined that feature selection would be useful for our project. (Kim et al., 2020) Forward and backward feature selection allow us to find the features with a p-value less than our significance level of 0.05 (95% confidence). With this method, our goal is to see which features are important to implement for success with our prediction model. 

##### PCA
We chose to use PCA to enhance variation and identify strong patterns in our dataset. We wanted to identify the first two principal components to get a better visual depiction of our data and see if there were any strong patterns identified by the first two principal components. We also wanted to use this type of dimensionality reduction because we had 15 features to begin with and wanted to identify how many features were needed to create a strong model. After running PCA, we chose to identify the explained variance ratio to understand if the two principal components would be enough to explain the variation in the model, or if we would need to  have a higher number of principal components. Our goal was to identify the number of principal components that would explain 95% of the variation in the data.

##### LASSO Regression
LASSO provides a simple way to reduce the number of features in a model. Using the Lasso method allows the inclusion of a penalty factor, usually selected through cross validation,which is used during the penalization of the L1 norm of the weights. Compared to stepwise feature selection, the LASSO method’s inclusion of cross validation helps assure that the generated model will respond well to more generalized future input. It is also useful in that it provides a concrete feature importance coefficient for each feature. These coefficients are useful in informing the decision about which features to include in the model.

#### Neural Network 
A multilayer perceptron allows us to incorporate all the parameters and features we deem potentially useful in our final predictions and "package" them as an input to the model. We chose to explore this option because it would allow us to mix and match different combinations of features and it has the capabilities to produce multidimensional outputs. In our case, the output dimension is two because we would like to predict both IMDB and Metascores. In addition, the nature of IMDB and Metascores, where they can easily be normalized to a range between 0 and 1, means that the Sigmoid activation function would be particularly effective
### Results and Discussion

#### Data Cleanup
The data clean up resulted in a dataset containing 6083 movies with 15 features. The features are title, original title, year, genre, duration, country of origin, language, IMDb score (avg_vote), Metascore, number of votes, budget, USA gross income, worldwide gross income, number of user reviews, and number of critic reviews.

#### Data Preprocessing

##### LASSO
LASSO selection was run on the preprocessed data with the goal of identifying important features to include in the data we run through our model. The input data for the LASSO selection model was first normalized, as were the output scores for both IMDb and Metascore. As shown in the figure below, for IMDb score, number of votes, duration, and budget were the three most important features.

![LassoIMDb](https://i.ibb.co/FXWM4vb/Lasso-IMDb.png)

The features selected by LASSO selection for Metascore were the same as those for IMDb, but the coefficients for the three categories were slightly higher than for IMDb. The normalized importances for each feature is shown in the graph below.

![LassoMetascore](https://i.ibb.co/FXWM4vb/Lasso-IMDb.png)

##### Feature selection
Forward and backward feature selection were run twice, once for the IMDb score and once for Metascore. Both targets returned the following features.

<img width="850" alt="imdb" src="https://user-images.githubusercontent.com/40035500/142087807-2060bd86-fa6c-427a-aacc-30d96601e10d.png">
<br/>

Looking at these results, it makes sense that both scores returned the same features as they should be impacted by the same information since they are both calculating how well a movie performed. It is also interesting that votes, duration, and budget are shown as 3 of the top 4 on both forward and backward feature selection. This leads us to believe that these are the most important features to consider when building our model. 

##### PCA
PCA was performed on the dataset to determine patterns in the data. First, we calculated the first and second principal components to visualize the data and identify any strong patterns. Reducing the number of features from 15 to 2 was helpful in visualizing our data, but the results determined that 2 features was not enough to retain a large portion of the variance in the data. Calculating the first and second principal components resulted in a very low explained variance, which indicated that only a small amount of variance was explained by the first two principal components.

![PCA_Visualization](https://i.ibb.co/9nmX0XS/PCA-Visualization.png)

The explained variance of the first two principal components was [0.37943203, 0.12747816]
The explained variance of the first principal component is below 40%, and the second is close to 10%, indicating that just the first two principal components do not retain much of the variance. While it was helpful to visualize our data, we decided that reducing the data to only the first two principal components would not be a useful model.

This indicated that a larger number of principal components were required for our data. We then chose to run PCA again, this time looking at only the first two principal components. We ran the model with the goal of explaining 95% of the variance. 

![ExpectedVariance_2PCs](https://i.ibb.co/VmpzQwt/Expected-Var-2-PCs.png)

![ExpectedVariance_0.95](https://i.ibb.co/QjXZRcH/Expected-Variance.png)
 
After performing PCA, the results determined that the first 7 principal components retained 95% of the variance in the data. In order to retain 95% of the variance, we need 7 features, which are aligned with our results from forward and backward feature selection.

The aforementioned feature selection methods were applied to the data passed into the neural network. The output of the LASSO, forward feature selection, and backward feature selection all indicated that duration, number of votes, and budget were important features to include in the model, so these features were initially passed into the neural network to provide the results below. The PCA analysis indicates that the data is fairly complex, as 7 principle components were required to retain 95% of the variance in the data. This information was also included when determining the number of features that should be passed into the neural network.

#### Neural Network
Another method we are using to approach this problem is to use a multilayer perceptron, or neural network, which takes in a tuple of parameters (e.g. budget, number of genres, number of votes) and outputs a tuple of predicted IMDB score and Metascore. However, for the sake of debugging and tuning, the model is currently only predicting IMDB scores for now.

To formulate the data into an x_train variable, we are normalizing everything using the MinMaxScaler function from sklearn.preprocessing, with the exception of IMDB and Metascores (used in y_train and y_test), as we simply normalized them by dividing all data points by 10 and 100 respectively because IMDB is scored up to 10 and Metascore is scored up to 100.

An example of our scaling method:

![MinMaxScaler Example](https://user-images.githubusercontent.com/41342635/142129599-353b8e47-0361-4427-b4b1-b9f65ee89e4f.png)

For the neural network, we are using Keras and Tensorflow, especially Keras’ Sequential model-building library. 
Here's our model:

![NN Model](https://user-images.githubusercontent.com/41342635/142129665-46745aeb-0ed9-480a-8ddc-ee3076b6fbbd.png)

This is our neural network as it currently stands, with two hidden layers and mainly sigmoid activation functions, since the final prediction would be scaled between 0 and 1. You can see a lot of comments demonstrating the many different loss functions, optimizers, and even additional dense layers we tried implementing to hopefully improve the model. We plan on adding more function arguments to the model, such as number_of_layers, activation_function, etc. that would allow us to quickly modify the model as needed without having to comment out code and replacing different lines, ultimately streamlining the tuning process.

The hyperparameters current stand as is because it produces the lowest loss of around 0.0025 when training finalizes.
Our training logs:
![Training Logs](https://user-images.githubusercontent.com/41342635/142129814-81afb21b-2638-4e79-a471-8924a72d6be6.png)

When comparing side-by-side the predicted vs actual values, the results are quite mixed:
![Predicted vs Actual](https://user-images.githubusercontent.com/41342635/142129911-7c5fed8a-a79f-43ee-8f01-63c27e277547.png)

We can see how some of the predictions come within a 0.01 error while others go to upwards of 0.15. This just shows that the model requires more hyper-parameter tuning before we would be able to see substantial results. After doing some more research, we believe that continuing to work on this model with the help of the KerasTuner library would be beneficial in helping us analyze exactly how many nodes per layer and how many hidden layers we would need. In regards to the activation, loss, and optimization functions, we are still in a trial stage and will have to do more research in that area to narrow down our options.



### References
Data: 
https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset 

Kharb L., Chahal D., Vagisha (2020) Forecasting Movie Rating Through Data Analytics. In: Batra U., Roy N., Panda B. (eds) Data Science and Analytics. REDSET 2019. Communications in Computer and Information Science, vol 1230. Springer, Singapore. https://doi.org/10.1007/978-981-15-5830-6_21
Hsu PY., Shen YH., Xie XA. (2014) Predicting Movies User Ratings with Imdb Attributes. In: Miao D., Pedrycz W., Ślȩzak D., Peters G., Hu Q., Wang R. (eds) Rough Sets and Knowledge Technology. RSKT 2014. Lecture Notes in Computer Science, vol 8818. Springer, Cham. https://doi.org/10.1007/978-3-319-11740-9_41
Basu S. (2019) Movie Rating Prediction System Based on Opinion Mining and Artificial Neural Networks. In: Kamal R., Henshaw M., Nair P. (eds) International Conference on Advanced Computing Networking and Informatics. Advances in Intelligent Systems and Computing, vol 870. Springer, Singapore. https://doi.org/10.1007/978-981-13-2673-8_6
Kim, Jong-Min & Xia, Leixin & Kim, Iksuk & Lee, Seungjoo & Lee, Keon-Hyung. (2020). Finding Nemo: Predicting Movie Performances by Machine Learning Methods. Journal of Risk and Financial Management. 13. 93. 10.3390/jrfm13050093.
