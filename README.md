# Machine-Learning

# Machine Learning Home Agriculture Monitoring System
We'll use TensorFlow 2.15.0 to create an anxiety detection model, training it with a labeled Twitter dataset, and convert it to TensorFlow Lite using post-training quantization.

The model uses a sequential architecture with an embedding layer to convert words into dense vectors. We'll start by training this custom neural network from scratch, including an embedding layer, a global average pooling layer, and dense layers for classification.

# Notebook ML 
[Fix notebook](https://colab.research.google.com/drive/1F-_sa02ywLOaSHNZCtbcthbz6BQWMXH9#scrollTo=ba10a13c-8d82-4977-bd67-a01cc70c87dd)

# Import the required libraries :
**Note**: This notebook requires TensorFlow 2.15.0 , and to run the provided code for creating and training the text classification model, as well as converting it to TensorFlow Lite with post-training quantization, you'll need the following libraries:

* Pandas
* Tensorflow 
* Numpy
* nltk
* re
* Sastrawi
* matplotlib
* wordcloud
* sci-kit learn
* imbalanced-learn

# Prepare the training data

* First let's download and organize the leaf disease dataset we'll use to retrain the model. Here we use an example of a leaf disease dataset image from Kaggle ( [Depression and Anxiety in Twitter (ID)]([https://www.kaggle.com/emmarex/plantdisease](https://www.kaggle.com/datasets/stevenhans/depression-and-anxiety-in-twitter-id 
))) 
 
 * We split the dataset into training and validation sets. We use 80% of the data for training and 20% for validation.
 
 * Tokenize the text data, considering the top 1000 most frequent words and replacing out-of-vocabulary words with a special token ("<OOV>"). Convert the tokenized text into padded sequences of uniform length (50 words).
 
 * Convert the labels into numerical values suitable for training the model.
  
# Build the model

We'll create a custom neural network for text classification. The model consists of an embedding layer, a global average pooling layer, and dense layers for classification.

# Create the base model

1. **Embedding Layer**: Converts each word index into a 16-dimensional vector.
2. **Global Average Pooling Layer**: Averages the embedding vectors, making the model invariant to the position of words.
3. **Dense Layers**: A fully connected layer with 24 neurons and ReLU activation, followed by an output layer with a single neuron and sigmoid activation for binary classification.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

# Configure the Model 
* Compile the model with binary cross-entropy loss and the Adam optimizer. The model is configured to evaluate performance based on accuracy.

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

# Train the Model 
* Now we can train the model using the training and validation data we prepared earlier by 10 epoch and 16 batch size. The result is shown below:

![Epoch](https://github.com/Equilibrare/Machine-Learning/assets/90241150/56cc7ac9-0050-4c42-bd56-a7d4debc7e00)

```python
history = model.fit(
    training_padded,
    training_labels,
    epochs=10,
    batch_size=16,
    validation_data=(validation_padded, validation_labels),
    verbose=2
)
```

# Converting the model to Tensorflow Lite


