import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from keras import activations
from keras.callbacks import EarlyStopping
from keras.layers import Dense,Bidirectional, LSTM, Attention
from keras.utils import to_categorical
from tensorflow.python.keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,roc_curve, auc
from sklearn.preprocessing import label_binarize
import time


def label_decoding(encoded_label):
    label = np.argmax(encoded_label, axis=1)
    return label


def label_encoding(labels):
    return to_categorical(labels)


def create_dataset(path):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    label_classes = {'Light1_ON': 1, 'Light1_OFF': 2, 'Light2_ON': 3, 'Light2_OFF': 4, 'Light3_ON': 5,
                     'Light3_OFF': 6, 'Fan_ON': 7, 'Fan_OFF': 8, 'Wrong_gestures': 9}

    data = []
    label = []

    is_read = None

    for dir1 in os.listdir(path):
        for image_name in os.listdir(os.path.join(path, dir1)):
            image_path = os.path.join(path, dir1, image_name)

            len = 0
            if (image_name != "Thumbs.db" and image_name != ".DS_Store"):
                image = cv2.imread(image_path)
                # print(image.shape)
                example = []
                imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                results = hands.process(imgRGB)
                if results.multi_hand_landmarks:
                    handlms = results.multi_hand_landmarks[0]
                    for id, ldm in enumerate(handlms.landmark):
                        example.append(ldm.x)
                        example.append(ldm.y)

                    data.append(example)
                    label.append(label_classes[dir1])
                    mpDraw.draw_landmarks(image, handlms, mpHands.HAND_CONNECTIONS)
                cv2.imshow('frame', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    return np.array(data), label_encoding(np.array(label))


def create_model():
        model = Sequential()
        model.add(LSTM(1000, return_sequences=True, activation='relu', input_shape=(1, 42)))
        model.add(LSTM(1000, return_sequences=True, activation='relu'))
        model.add(LSTM(500, return_sequences=False, activation='relu'))
        model.add(Dense(units=2000, activation='relu'))
        model.add(Dense(units=1000, activation='relu'))
        model.add(Dense(units=10, activation=activations.softmax))
        return model

model = create_model()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
history=model.fit(x_train, y_train, epochs=300, batch_size=50,validation_data=(x_val, y_val))




def confusion_metrics(actual, predict):
    confusion_matrix = metrics.confusion_matrix(actual, predict)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                display_labels=['Light1_ON', 'Light1_OFF', 'Light2_ON', 'Light2_OFF',
                                                                'Light3_ON', 'Light3_OFF', 'Fan_ON', 'Fan_OFF',
                                                                'Wrong_gestures'])

    cm_display.plot()
    plt.show()

#trainX, trainY = create_dataset(path)
#path = r"C:\Users\zkh\Desktop\Dataset for training\dataset\without_ldm"
#trainX, trainY = create_dataset(path)
trainX=np.load(r"C:\Users\zkh\Desktop\myproject\Graph\trainX.npy")
trainY=np.load(r"C:\Users\zkh\Desktop\myproject\Graph\trainY.npy")
print("trainx shape:", trainX.shape)
print("trainy shape:", trainY.shape)

x_train, x_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.1, random_state=0)
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1)
#x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
#x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
#x_val = np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1]))

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

model = create_model()
print("ok")

callback = EarlyStopping(monitor = 'loss', patience = 3, mode = 'min', restore_best_weights = True)
print(callback)

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
history=model.fit(x_train, y_train, epochs=300, batch_size=50,validation_data=(x_val, y_val))
#history=model.fit(x_train, y_train, epochs =300, batch_size=32,callbacks=[callback],validation_data=(x_val, y_val))
model.save("LSTM_hand_gesture_model")
x_predicted = model.predict(x_test)
# print("predicted values: ", x_predicted)
x_predicted = np.argmax(x_predicted, axis=1)
y_test = np.argmax(y_test, axis=1)
print("predicted values: ", x_predicted)
print("actual values: ", y_test)

print("test accuracy:",metrics.accuracy_score(y_test, x_predicted))
confusion_metrics(y_test, x_predicted)


# Plot the training loss curve and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.show()

precision_scores = []
recall_scores = []
f1_scores = []

# Loop through epochs
for i in range(len(history.history['loss'])):
    # Get predictions on the test set for the current epoch
    current_predictions = model.predict(x_test)
    current_predictions = np.argmax(current_predictions, axis=1)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, current_predictions, average='weighted', zero_division=1)

    sensitivity = recall
    specificity = 1 - sensitivity
    # Store scores
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(fscore)



# Plot all metrics in one graph
plt.figure(figsize=(10, 6))
plt.plot(precision_scores, label='Precision', color='blue', linestyle='-', marker='o')
plt.plot(recall_scores, label='Recall', color='green', linestyle='-', marker='o')
plt.plot(f1_scores, label='F1-Score', color='red', linestyle='-', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Model Metrics Over Epochs')
plt.legend()
plt.show()
print(f' Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {fscore:.4f}, Sensitivity:{sensitivity:.4f},Specificity:{specificity:.4f}')


#Roc Curve
# Assuming you have the true labels in y_test and predicted probabilities in x_predicted
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

n_classes = 10  # Number of classes in your problem

# Assuming you have the true labels in y_test and predicted probabilities in x_predicted
y_test_bin = label_binarize(y_test, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_predicted_bin = label_binarize(x_predicted, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], x_predicted_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), x_predicted_bin.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves for each class
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'pink', 'brown', 'gray', 'olive'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i + 1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Multi-Class')
plt.legend(loc="lower right")
plt.show()

# Calculate average AUC across classes
average_auc = sum(roc_auc.values()) / n_classes
print("Average AUC:", average_auc)


#3D Scatter plot of actual and predicted
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Sample data for actual and predicted labels
actual = y_test  # Replace actual_6 with your actual labels array
predicted = x_predicted  # Replace predicted_6 with your predicted labels array

# Assign arbitrary x, y, and z values to each data point
x_values = np.random.rand(len(actual))
y_values = np.random.rand(len(predicted))
z_values = np.random.rand(len(actual))

# Create a figure and a 3D subplot with a white background
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')  # Set background color

# Plot the data points with colors based on classification
for i in range(len(actual)):
    color = 'blue' if actual[i] == predicted[i] else 'red'
    ax.scatter(x_values[i], y_values[i], z_values[i], c=color, s=50, edgecolor='k', alpha=0.7)

# Customize the grid lines
ax.grid(True, linestyle='--', alpha=0.5)

# Add labels with larger fonts

# Set the title
plt.title('3D Scatter Plot', fontsize=16)

# Adjust the view angle for a better perspective
ax.view_init(elev=20, azim=30)

# Add a legend
blue_patch = plt.Line2D([0], [0], marker='o', color='b', markersize=8, label='Actual == Predicted', linestyle='None')
red_patch = plt.Line2D([0], [0], marker='o', color='r', markersize=8, label='Actual != Predicted', linestyle='None')
plt.legend(handles=[blue_patch, red_patch], loc='best')

# Show the plot
plt.tight_layout()
plt.show()
