**Name**:Bavirisetti Pavani Durga Bhavani
**Company**:CODTECH IT Solutions
**ID**:CT08DS4255
**Domain**:Machine Learning
**Duration**:July to August 2024
**Mentor**:Neela Santhosh Kumar

##Overview Of The Project
### Project:Sentimental Analysis on Movie Reviews 
![Screenshot (72)](https://github.com/user-attachments/assets/61998f88-a4d5-4f15-90ab-c9570409418b)

##Objective:
The main objective of this task is to develop a machine learning model that can accurately classify movie reviews as either positive or negative. This involves using natural language processing (NLP) techniques to analyze the sentiment expressed in the text of movie reviews.

###Key Activities:
-**Data Collection**:Load the IMDb Movie Reviews dataset using tensorflow.keras.datasets.imdb.
-**Data Preprocessing**:Limit the dataset to the top 10,000 most frequent words (num_words=10000).Pad sequences to ensure all reviews have the same length (maxlen=256) using tensorflow.keras.preprocessing.sequence.pad_sequences.
-**Model Building**:Create a Sequential model using tensorflow.keras.models.Sequential.
Add an Embedding layer with input_dim=10000, output_dim=32, and input_length=256 to convert word indices into dense vectors of fixed size.
Add a Flatten layer to convert the 2D matrix of embeddings into a 1D vector.
Add a Dense layer with 64 units and ReLU activation for a fully connected layer.
Add a Dropout layer with a rate of 0.5 to prevent overfitting.
Add a Dense layer with 1 unit and sigmoid activation for binary classification.
-**Model Compilation**:Compile the model using the Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric.
-**Model Training**:Train the model on the training data for 10 epochs with a batch size of 512 and a validation split of 0.2 using the fit method.
-**Model Evaluation**:Evaluate the model on the test data using the evaluate method and print the test accuracy.
-**Results Visualization**:Plot training and validation accuracy and loss values over epochs using Matplotlib.
-**Word Index Adjustment**:Get the word index from the IMDb dataset and adjust indices to account for reserved tokens (e.g., <PAD>, <START>, <UNK>, <UNUSED>).
-**Example Review Processing**:Encode a sample review text into word indices using the adjusted word index.
Pad the encoded review sequence to match the input length of the model.
-**Sentiment Prediction**:Use the trained model to predict the sentiment of the preprocessed review and print whether the sentiment is positive or negative.
## Features
- **Binary Classification:** The model classifies movie reviews into positive or negative categories.
- **Text Preprocessing:** Handles data cleaning, tokenization, and padding of text sequences.
- **Model Training:** Utilizes an embedding layer for word representation, followed by dense layers for classification.
- **Visualization:** Plots the training and validation accuracy and loss over epochs.

###Technologies Used:
-**Programming Language**:Python
-**Libraries and Frameworks**:
TensorFlow/Keras: For building, training, and evaluating the sentiment analysis model.
NumPy: For numerical operations.
pandas: For data manipulation and analysis.
Matplotlib: For plotting training and validation accuracy and loss values.
IMDb Dataset: For training and testing the sentiment analysis model.


