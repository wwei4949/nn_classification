import numpy as np
import time
import matplotlib.pyplot as plt

# Load data
train_images_np = np.load('./Project3_Data/MNIST_train_images.npy')
train_labels_np = np.load('./Project3_Data/MNIST_train_labels.npy')
val_images_np = np.load('./Project3_Data/MNIST_val_images.npy')
val_labels_np = np.load('./Project3_Data/MNIST_val_labels.npy')
test_images_np = np.load('./Project3_Data/MNIST_test_images.npy')
test_labels_np = np.load('./Project3_Data/MNIST_test_labels.npy')

# Define helper functions
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def CrossEntropy(y_hat, y):
    return -np.sum(y * np.log(y_hat))

def one_hot(Y):
    return np.eye(10)[Y]

# Define MLP class
class MLP():

    def __init__(self):
        self.W1 = np.random.normal(0, 0.1, (64, 784)) / np.sqrt(784)
        self.b1 = np.zeros(64)
        self.W2 = np.random.normal(0, 0.1, (10, 64)) / np.sqrt(64)
        self.b2 = np.zeros(10)
        self.reset_grad()

    def reset_grad(self):
        self.W2_grad = 0
        self.b2_grad = 0
        self.W1_grad = 0
        self.b1_grad = 0

    def forward(self, x):
        self.x = x
        self.W1x = np.dot(x, self.W1.T)
        self.a1 = self.W1x + self.b1

        self.f1 = sigmoid(self.a1)
        self.W2x = np.dot(self.f1, self.W2.T)
        self.a2 = self.W2x + self.b2
        self.y_hat = softmax(self.a2)
        return self.y_hat

    def update_grad(self, y):
        # Compute the gradients for the current observation y and add it to the gradient estimate over the entire batch
        # Uncomment and complete the following lines
        dA2dW2 = self.f1
        dA2dF1 = self.W2
        dF1dA1 = self.f1 * (1 - self.f1)
        dA1dW1 = self.x

        batch_size = y.shape[0]
        l2_lambda = 0.0001
        dLdA2 = self.y_hat - y
        dLdW2 = np.dot(dLdA2.T, dA2dW2) / batch_size
        dLdW2 = dLdW2 + l2_lambda * self.W2

        dLdb2 = np.sum(dLdA2) / batch_size
        dLdF1 = np.dot(dLdA2, dA2dF1)
        dLdA1 = dLdF1 * dF1dA1
        dLdW1 = np.dot(dLdA1.T, dA1dW1) / batch_size
        dLdW1 = dLdW1 + l2_lambda * self.W1

        dLdb1 = np.sum(dLdA1) / batch_size

        self.W2_grad = self.W2_grad + dLdW2
        self.b2_grad = self.b2_grad + dLdb2
        self.W1_grad = self.W1_grad + dLdW1
        self.b1_grad = self.b1_grad + dLdb1

    def update_params(self, learning_rate):
        self.W2 = self.W2 - learning_rate * self.W2_grad
        self.b2 = self.b2 - learning_rate * self.b2_grad.reshape(-1)
        self.W1 = self.W1 - learning_rate * self.W1_grad
        self.b1 = self.b1 - learning_rate * self.b1_grad.reshape(-1)

myNet = MLP()

# Configured code for trained weights

try:
    with open('MLP_weights.npz', 'rb') as f:
        weights = np.load('MLP_weights.npz')
        myNet.W1 = weights['W1']
        myNet.b1 = weights['b1']
        myNet.W2 = weights['W2']
        myNet.b2 = weights['b2']

        y_hat_test = myNet.forward(test_images_np)
        test_pred = np.argmax(y_hat_test, axis=1)
        test_acc = np.mean(test_pred == test_labels_np)
        test_accuracy = np.mean(test_pred == test_labels_np)
        print(f'MLP Test Accuracy: {test_accuracy}')

except FileNotFoundError:
    # Forward pass
    num_samples = 5

    # predictions for the first 5 images in the test set after one forward pass
    y_pred = myNet.forward(test_images_np[:num_samples])

    # convert predictions and true labels to class indices
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = test_labels_np[:num_samples]

    # plot the images and their corresponding labels
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(test_images_np[i].reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {y_pred_classes[i]}\nActual: {y_true_classes[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Backpropagation
    n_epochs = 100
    batch_size = 256
    learning_rate = 1e-3

    myNet = MLP()
    accuracy = []

    for epoch in range(n_epochs):
        adaptive_learning_rate = learning_rate * (0.97 ** epoch)
        for i in range(0, train_images_np.shape[0], batch_size):
            batch_images = train_images_np[i:i + batch_size]
            batch_labels = train_labels_np[i:i + batch_size]
            myNet.reset_grad()
            y_hat = myNet.forward(batch_images)
            myNet.update_grad(one_hot(batch_labels))
            myNet.update_params(adaptive_learning_rate)

        y_hat_train = myNet.forward(train_images_np)
        train_pred = np.argmax(y_hat_train, axis=1)
        train_acc = np.mean(train_pred == train_labels_np)
        accuracy.append(train_acc)

    plt.plot(accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs trained on 50000 Images')
    plt.show()


    # training and validation accuracies

    learning_rate = 1e-2
    n_epochs = 70
    batch_size = 64

    # first 2000 images
    MyNet = MLP()
    two_thousand_train_images = train_images_np[:2000]
    two_thousand_train_labels = train_labels_np[:2000]

    two_thousand_train_accuracy = []
    val_accuracy = []

    start_time = time.time()

    for epoch in range(n_epochs):
        adaptive_learning_rate = learning_rate * (0.97 ** epoch)
        for i in range(0, two_thousand_train_images.shape[0], batch_size):
            batch_images = two_thousand_train_images[i:i + batch_size]
            batch_labels = two_thousand_train_labels[i:i + batch_size]

            myNet.reset_grad()
            y_hat = myNet.forward(batch_images)
            myNet.update_grad(one_hot(batch_labels))
            myNet.update_params(adaptive_learning_rate)

        y_hat_train = myNet.forward(two_thousand_train_images)
        train_pred = np.argmax(y_hat_train, axis=1)
        train_acc = np.mean(train_pred == two_thousand_train_labels)
        two_thousand_train_accuracy.append(train_acc)

        y_hat_val = myNet.forward(val_images_np)
        val_pred = np.argmax(y_hat_val, axis=1)
        val_acc = np.mean(val_pred == val_labels_np)
        val_accuracy.append(val_acc)

    elapsed_time = time.time() - start_time
    print(f'Training time for 2000 images: {elapsed_time:.2f} seconds')

    plt.plot(two_thousand_train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epochs Trained on 2000 Images')
    plt.show()

    # all training images

    myNet = MLP()
    fifty_thousand_train_accuracy = []
    fifty_thousand_val_accuracy = []
    val_losses = []

    start_time = time.time()

    for epoch in range(n_epochs):
        adaptive_learning_rate = learning_rate * (0.97 ** epoch)
        for i in range(0, train_images_np.shape[0], batch_size):
            batch_images = train_images_np[i:i + batch_size]
            batch_labels = train_labels_np[i:i + batch_size]

            myNet.reset_grad()
            y_hat = myNet.forward(batch_images)
            myNet.update_grad(one_hot(batch_labels))
            myNet.update_params(adaptive_learning_rate)

        y_hat_train = myNet.forward(train_images_np)
        train_pred = np.argmax(y_hat_train, axis=1)
        train_acc = np.mean(train_pred == train_labels_np)
        fifty_thousand_train_accuracy.append(train_acc)


        y_hat_val = myNet.forward(val_images_np)
        val_pred = np.argmax(y_hat_val, axis=1)
        val_acc = np.mean(val_pred == val_labels_np)
        fifty_thousand_val_accuracy.append(val_acc)

    elapsed_time = time.time() - start_time
    print(f'Training time for 50000 images: {elapsed_time:.2f} seconds')

    plt.plot(fifty_thousand_train_accuracy, label='Training Accuracy')
    plt.plot(fifty_thousand_val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epochs trained on 50000 Images')
    plt.show()

    #Testing Accuracy

    y_hat_test = myNet.forward(test_images_np)
    test_pred = np.argmax(y_hat_test, axis=1)
    test_acc = np.mean(test_pred == test_labels_np)
    test_accuracy = np.mean(test_pred == test_labels_np)

    print(f'MLP Test Accuracy: {test_accuracy}')

    # Saving Weights
    np.savez('MLP_weights.npz', W1=myNet.W1, b1=myNet.b1, W2=myNet.W2, b2=myNet.b2)

    # Confusion Matrix
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for i in range(len(test_labels_np)):
        confusion_matrix[test_labels_np[i]][test_pred[i]] += 1
    confusion_matrix = 100 * confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
    print(confusion_matrix)

    # Visualize Weights
    num_templates = myNet.W1.shape[0]
    ncols = 8
    nrows = num_templates // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            template = myNet.W1[index].reshape(28, 28)
            axes[i, j].imshow(template, cmap='binary')
            axes[i, j].axis('off')

    plt.show()

## Template for ConvNet Code
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    #From https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        p = 0.5
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.view(-1,1,28,28))))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # first dropout mask
        x = F.relu(self.fc2(x))
        x = self.dropout(x) # second dropout mask
        x = self.fc3(x)
        return x

#Your training and testing code goes here

# Preprocess the data
train_images_np = (train_images_np / 255.0 - 0.1307) / 0.3081
train_images = torch.tensor(train_images_np, dtype=torch.float32)
train_labels = torch.tensor(train_labels_np, dtype=torch.long)

val_images_np = (val_images_np / 255.0 - 0.1307) / 0.3081
val_images = torch.tensor(val_images_np, dtype=torch.float32)
val_labels = torch.tensor(val_labels_np, dtype=torch.long)


test_images_np = (test_images_np / 255.0 - 0.1307) / 0.3081
test_images = torch.tensor(test_images_np, dtype=torch.float32)
test_labels = torch.tensor(test_labels_np, dtype=torch.long)

test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

# Initialize and train the network

net = ConvNet()

try:
    with open('CNN_weights.npz', 'rb') as f:
        net = ConvNet()
        net.load_state_dict(torch.load('CNN_weights.npz'))
        net.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"CNN Test Accuracy: {correct / total}%")

except FileNotFoundError:

    def train_and_plot(train_size, b_s, l_r, epochs, val):
        train_subset = train_images[:train_size]
        train_labels_subset = train_labels[:train_size]
        train_subset_dataset = torch.utils.data.TensorDataset(train_subset, train_labels_subset)
        train_subset_loader = torch.utils.data.DataLoader(train_subset_dataset, batch_size=b_s, shuffle=False)

        val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=b_s, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=l_r, momentum=0.9, weight_decay=1e-4)
        train_accuracy = []
        val_accuracy = []

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_subset_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_correct = 0
            train_total = 0
            with torch.no_grad():
                for data in train_subset_loader:
                    images, labels = data
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
            train_accuracy.append(train_correct / train_total)
            if val:
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for data in val_loader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                val_accuracy.append(val_correct / val_total)
        plt.plot(train_accuracy, label="Training")
        if val:
            plt.plot(val_accuracy, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title(f"Accuracy vs Epoch for {train_size} Training Images")
        plt.show()

    train_and_plot(50000, 256, 0.001, 10, False)

    net = ConvNet()
    start_time = time.time()
    train_and_plot(2000, 64, 0.01, 10, True)
    elapsed_time = time.time() - start_time
    print(f'Training time for 2000 images: {elapsed_time:.2f} seconds')


    net = ConvNet()
    start_time = time.time()
    train_and_plot(50000, 64, 0.01, 10, True)
    elapsed_time = time.time() - start_time
    print(f'Training time for 50000 images: {elapsed_time:.2f} seconds')

    # Saving the model
    torch.save(net.state_dict(), 'CNN_weights.npz')

    # Testing Accuracy

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"CNN Test Accuracy: {correct / total}")