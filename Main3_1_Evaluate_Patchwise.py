from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import load_model
from sklearn.metrics import roc_curve

######################### Step1:  Initialize the parameters and file paths #########################
# configure the parameters
# configure the parameters
batch_size = 32
num_classes = 2
epochs = 1
image_height = 72
image_width = 72

# Set the corresponding file paths
model_folder = 'VGG16_NT_Fusion_Model'
# model_folder = 'VGG19_NT_Fusion_Model'
# Configure the train, val, and test p
base_dir = 'D:/Data/Image/Biomedicine/integrated/MIAS_Patches/MIAS_B_M_Norm_Preprocess/MIAS_NT_ForCNN'
test_dir = os.path.join(base_dir, 'test_image')

######################### Step2:  Obtain the test dataflow #########################
test_datagen = ImageDataGenerator(
    samplewise_center=True
    # rescale=1./255
)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_height, image_width),
        batch_size=1,
        shuffle=False,
        class_mode='categorical')

######################### Step3:  Load the best model trained before and evaluate its performance #########################
# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), model_folder, 'checkpoints')
# load the best model
# If the error appears 'Error in loading the saved optimizer', it doesn't matter!
best_model = load_model(os.path.join(checkpoint_dir, "weights-improvement-best.hdf5"))

######################### Step3.1:  Obtain the loss and acc
# Score trained best_model.
print('The evaluation starts!\n')
scores = best_model.evaluate_generator(
    test_generator,
    steps=len(test_generator))

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

######################### Step3.2:  Obtain the precision 、recall 、f1-score
# Obtain the prediction
pred_prob = best_model.predict_generator(
    test_generator,
    steps=len(test_generator))

# The index for prediction of testing set
pred_labels = np.argmax(pred_prob, axis=1)
# The scores for prediction of testing set
pred_scores = np.amax(pred_prob, axis=1)

# The true labels of testing set
true_labels = test_generator.classes

# python 预测测试集报告 precision 、recall 、f1-score 、support
# https://blog.csdn.net/weixin_42342968/article/details/83617607
cfm = confusion_matrix(true_labels, pred_labels)
print(classification_report(true_labels, pred_labels, digits=4))

######################### Step3.3: Plot the roc curve for each class
#Area under the ROC curve
fpr, tpr, thresholds = roc_curve((true_labels), pred_scores)
AUC_ROC = roc_auc_score(true_labels, pred_scores)
lw = 2
# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
print("\nArea under the ROC curve: " +str(AUC_ROC))
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(true_labels, pred_scores)
precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision,recall)
print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")








