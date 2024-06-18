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

# Configure the model :

Although this method is called compile(), it's basically a configuration step that's required before we can start training. And because the majority of the model graph is frozen in the base model, weights from only the last convolution and dense layers are trainable.

# Train the model

Now we can train the model using data provided by the train_generator and val_generator that we created at the beginning.
![image](https://user-images.githubusercontent.com/67249292/120916823-f07d2880-c6d5-11eb-8635-c5323aca9741.png)

# Fine tune the base model

So far, we've only trained the classification layers—the weights of the pre-trained network were not changed.

One way we can increase the accuracy is to train (or "fine-tune") more layers from the pre-trained model. That is, we'll un-freeze some layers from the base model and adjust those weights (which were originally trained with 1,000 ImageNet classes) so they're better tuned for features found in our leaf disease dataset.

# Un-freeze more layers
So instead of freezing the entire base model, we'll freeze individual layers.

# Reconfigure the model
Now configure the model again, but this time with a lower learning rate (the default is 0.001).

# Continue training
Now let's fine-tune all trainable layers. This starts with the weights we already trained in the classification layers, so we don't need as many epochs.

Our model better, but it's not ideal.

The validation loss is still higher than the training loss, so there could be some overfitting during training. The overfitting might also be because the new training set is relatively small with less intra-class variance, compared to the original ImageNet dataset used to train [MobileNet V2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2).

So this model isn't trained to an accuracy that's production ready, but it works well enough as a demonstration.

# Save all the models that have been trained
You can check here to see all the models that have been built [model_project](https://github.com/maulanaakbardj/Home-Agriculture-Monitoring-System/tree/main/ML/model_project)

# Convert Model to TFLite
Ordinarily, creating a TensorFlow Lite model is just a few lines of code with [TFLiteConverter](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter). 

However, this .tflite file still uses floating-point values for the parameter data, and we need to fully quantize the model to int8 format.

To fully quantize the model, we need to perform [post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) with a representative dataset, which requires a few more arguments for the TFLiteConverter, and a function that builds a dataset that's representative of the training dataset.

You can check here to see [result TFLite](https://github.com/maulanaakbardj/Home-Agriculture-Monitoring-System/tree/main/ML/TFLite)

# Compare the accuracy
We have a fully quantized TensorFlow Lite model. To be sure the conversion went well, so we compare it.

But again, these results are not ideal but better for prototype.


