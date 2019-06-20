# IMDB-Sentiment-Analysis-using-Neural-Network
The objective of this task is to perform Sentiment Analysis of IMDB Movie Reviews using LSTM.

![alt text](https://github.com/shreyaswankhede/IMDB-Sentiment-Analysis-using-Neural-Network/blob/master/10.PNG
 "Correlation between features")

***

# Sentiment Analysis:

* Sentiment analysis and Opinion mining is the computational study of User opinion to analyze the social, psychological, philosophical, behavior and perception of an individual person or a group of people about a product, policy, services and specific situations using Machine learning technique. Sentiment analysis is an important research area that identifies the people’s sentiment underlying a text and helps in decision making about the product. Sentiment Analysis is one of the important applications in Natural Language Processing(NLP).

***

# Data

* To perform Sentiment Analysis in python I have used Keras, which is an open-source neural-network library. Keras has different builtin dataset to perform different machine learning task. To know more dataset in Keras go to the link https://keras.io/datasets/. I have selected 'IMDB Movie reviews sentiment classification' dataset from Keras.

# Steps:

1. Loading training and test data.
2. Checking dataset with 25000 training samples, 25000 test samples.
3. Reviews are stored as a sequence of word IDs in interger format. 
4. Mapping the reviews to the original words.
5. Checking Maximum review length and minimum review length.
6. RNN model for sentiment analysis
7. Padding
8. Evaluation and Model Training. 

***

# Model Summary

![alt text](https://github.com/shreyaswankhede/IMDB-Sentiment-Analysis-using-Neural-Network/blob/master/11.PNG
 "Correlation between features")

# Training the model by setting batch size & epochs:

## 1. batch_size = 512 & num_epochs = 5

* Training and validation loss

![alt text](https://github.com/shreyaswankhede/IMDB-Sentiment-Analysis-using-Neural-Network/blob/master/12.PNG
 "Correlation between features")
 
 * Training and validation accuracy
 
 ![alt text](https://github.com/shreyaswankhede/IMDB-Sentiment-Analysis-using-Neural-Network/blob/master/13.PNG
 "Correlation between features")
 
 ## 2. batch_size = 64 & num_epochs = 3

* Training and validation loss

![alt text](https://github.com/shreyaswankhede/IMDB-Sentiment-Analysis-using-Neural-Network/blob/master/14.PNG
 "Correlation between features")
 
 * Training and validation accuracy
 
 ![alt text](https://github.com/shreyaswankhede/IMDB-Sentiment-Analysis-using-Neural-Network/blob/master/15.PNG
 "Correlation between features")

***

# Results:

There are different ways in which we can build our model by improving the accuracy of our model by experimenting with different architectures, layers and parameters.

In this example, 
<br> Accuracy for model 1 = 84%
<br> Accuracy for model 2 = 86%

***

<br>Thank You!	
<p><!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/shreyaswankhede" aria-label="Follow @shreyaswankhede on GitHub">Follow @shreyaswankhede</a>
