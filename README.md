# Artificial Intelligence Project

## Goal
The goal of this project is to explore various artificial intelligence techniques, including deep learning and machine learning, through practical assignments. Each assignment aims to solve specific problems using different datasets and neural network architectures.

## Libraries Used
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- spaCy

## Key Points
- Loading and preprocessing datasets
- Building and training different neural network architectures
- Implementing custom dataset classes
- Using GPU for training and comparing with CPU
- Handling overfitting using techniques like dropout
- Evaluating model performance with various metrics
- Utilizing transfer learning with pre-trained models

## Expected Outcomes
- Understanding of how to load and preprocess different types of data
- Ability to build and train neural networks from scratch
- Knowledge of training on GPU and performance comparison
- Skills in using advanced neural network architectures
- Experience in evaluating and improving model performance

---

## Lab 1

### Task
1. Load the MNIST dataset into train and test data loaders.
2. Create a feedforward neural network:
   - Adjust the input layer for MNIST data.
   - Use `log_softmax` as the output layer activation function.
   - Use `ReLU` as the hidden layer activation function.
3. Train the network on the training data for 50 epochs using the negative log-likelihood loss.
4. Test the network on the MNIST test data and output accuracy and loss.
5. Compare training times between CPU and GPU.

---

## Lab 2

### Task
1. Write a custom dataset class for the Titanic data using selected features.
2. Build a neural network with one hidden layer of size 3 and use BCE loss.
3. Evaluate model performance on validation data using accuracy.
4. Plot training and validation performance.
5. Increase network complexity to overfit on training data.
6. Introduce a dropout layer to mitigate overfitting.

---

## Lab 3

### Task
1. Train the network from the PyTorch CNN tutorial.
2. Modify the network architecture and train it:
   - Conv layers with varying depths and ReLU activations.
   - Max pooling layers.
   - Fully connected layers.
3. Compare training times on GPU and CPU.
4. Log training loss in TensorBoard.
5. Modify test metric to consider the top three outputs.
6. Plot misclassified examples in TensorBoard.

---

## Lab 4

### Task
1. Choose a dataset from torchvision (excluding CIFAR, ImageNet, FashionMNIST, or eMNIST).
2. Display example images and print dataset size.
3. Design a CNN with batch normalization layers and train it.
4. Measure accuracy on holdout test data.
5. Use transfer learning with ResNet18 and EfficientNet_B5.
6. Compare accuracy and training times for different approaches.

---

## Lab 5

### Task
1. Train a text classification model on the TweetEval emotion recognition dataset using LSTMs and GRUs.
   - Use the last output of the LSTM in the loss function.
   - Set embedding dim to 128 and hidden dim to 256.
   - Use spaCy for tokenization and limit vocabulary to 5000 words.
2. Evaluate accuracy on the test set.
3. Compare training with LSTMs and GRUs.
4. (Bonus) Implement batch training with padding and compare training times.

---

## Conclusion
Through these assignments, you will gain hands-on experience with various AI techniques, deepen your understanding of neural networks, and improve your ability to apply machine learning methods to real-world problems.
