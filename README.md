# Introduction

## Problem Statement

Given a dataset of movie reviews from Rotten Tomatoes, the objective is to perform sentiment analysis and classify each review into one of the following categories: rotten or fresh. The sentiment classification is based on the text of the review that is classified into one of the following categories: positive, negative, or neutral as well as the type of critic (Top Critics or All Critics).

The goal of this sentiment analysis is to provide insights into the overall sentiment of the movie reviews, identify the most positive and negative movies, and potentially improve the quality of movies by analyzing the feedback provided by the reviewers.

## Approach

To achieve this objective, we need to preprocess the text data, extract features, train a sentiment classification model using machine learning techniques, and evaluate the performance of the model. Additionally, we may need to explore the dataset to identify potential issues such as class imbalance or missing data and address them accordingly. Finally, we can visualize the results of the sentiment analysis to gain insights into the sentiment of the movie reviews and potentially make recommendations to improve the quality of the movies.

## Data

The website "Rotten Tomatoes" allows to compare the ratings given by regular users (audience score) and the ratings given/reviews provided by critics (tomatometer) who are certified members of various writing guilds or film critic-associations.

We have two datasets, namely critics dataset and movies dataset.

In the movies dataset each record represents a movie available on Rotten Tomatoes, with the URL used for the scraping, movie tile, description, genres, duration, director, actors, users' ratings, and critics' ratings.

In the critics dataset each record represents a critic review published on Rotten Tomatoes, with the URL used for the scraping, critic name, review publication, date, score, and content.

Link to Dataset: https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset

## Project Members:
* HACHEM Racha
* FESTA Denis
* BHATT Ragi



**Strategy 1**: Following this path we created two balanced datasets to tackle two different problems:
- prediction of the type of the review (classification).
- prediction of the score of the review (regression).

Both the tasks were approached working on top of a pretrained BERT model,
stacking some layers upon it:
- in the case of classification, the last layer is a linear layer with two neurons as output, that is, the two probabilities that the review is either 
- in the case of regression, the last layer is a linear layer with one neuron as output (the actual predicted score, a number between 0 and 1 that will
be rescaled accordingly),
- in both cases a dropout layer was added just after the pre-trained BERT model output, and an additional linear hidden layer was added.

Accuracy: Classification + Bert = 81 %

*It might be interesting to see how the Bert model would perform on the entire dataset, or even on a larger segment of the data.*



**Strategy 2**: In this approach, we used a simple method (SentimentIntensityAnalyzer) to analyze the sentiment behind the review. The SentimentIntensityAnalyzer gives the results in a format of multiple scores: negative, neutral, positive, and compound. After getting the scores of all the reviews, we labeled the prediction with positive and egative based on the compound score (a score lower than 0 is negative and a score greater or equal to 0 is positive). 
Next, to be able to visualize what the algorithm did, we plotted the word cloud of the positive and negative vocabulary.
We also calculated the accuracy score by comparing the predicted sentiment with the true label (positive = fresh, and negative = rotten) 
Lastly, to be able to implement such approach in a real-life scenario we seperately showed the reviews of the top critics and normal critics. This approach can be used to accurately and reliably rate movies based on reviews from true top critics

Accuracy: Classification + Sentiment Analysis = 64.7%

#Conclusion:
##What Worked:
Word Cloud for Positive Reviews
![image](https://github.com/RagiBhatt07/Sentiment-Analysis-Rotten-Tomato/assets/124009502/222023b2-2898-48ce-af38-4357f92f3375)


Word Cloud for Negative Reviews
![image](https://github.com/RagiBhatt07/Sentiment-Analysis-Rotten-Tomato/assets/124009502/b76ca871-baed-4b2a-8d0a-7d1d4d19f9dd)


*   Balancing the training data
*   Selection of Important features from the dataset
* Using Bert model did improve the performnace of the model.
*   The approach with considering only the review type showed better results than the other.

##Limitations:



*   Preprocessing might have led to overfitting the model
*   Could not test the first approach on a bigger dataset or the entire dataset




##What could have been done better?



*   Feature engineering to improve the accuracy of sentiment classification. For example, features like word count, sentence length, and punctuation count could be used.


*  Providing pre-defined lists of words and phrases that are associated with positive and negative sentiment in the context of movie reviews to improve sentiment classification accuracy













