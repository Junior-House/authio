# Authio

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 64)                2048
_________________________________________________________________
batch_normalization_1 (Batch (None, 64)                256
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0
_________________________________________________________________
dense_2 (Dense)              (None, 64)                4160      
_________________________________________________________________
batch_normalization_2 (Batch (None, 64)                256
_________________________________________________________________
dropout_2 (Dropout)          (None, 64)                0
_________________________________________________________________
dense_3 (Dense)              (None, 64)                4160
_________________________________________________________________
batch_normalization_3 (Batch (None, 64)                256
_________________________________________________________________
dropout_3 (Dropout)          (None, 64)                0
_________________________________________________________________
dense_4 (Dense)              (None, 32)                2080
_________________________________________________________________
batch_normalization_4 (Batch (None, 32)                128
_________________________________________________________________
dropout_4 (Dropout)          (None, 32)                0
_________________________________________________________________
dense_5 (Dense)              (None, 16)                528
_________________________________________________________________
batch_normalization_5 (Batch (None, 16)                64
_________________________________________________________________
dropout_5 (Dropout)          (None, 16)                0
_________________________________________________________________
dense_6 (Dense)              (None, 2)                 34
=================================================================
Total params: 13,970
Trainable params: 13,490
Non-trainable params: 480
_________________________________________________________________
tracking <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=0> tp
Train on 2560 samples, validate on 640 samples
Epoch 1/30
2019-10-31 14:17:44.359883: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_100.dll
2560/2560 [==============================] - 2s 640us/step - loss: 0.3229 - accuracy: 0.8656 - true_positive: 540.4375 - val_loss: 0.4399 - val_accuracy: 0.7703 - val_true_positive: 91.2500
Epoch 2/30
2560/2560 [==============================] - 1s 233us/step - loss: 0.2079 - accuracy: 0.9184 - true_positive: 583.3000 - val_loss: 0.2665 - val_accuracy: 0.8734 - val_true_positive: 125.3500
Epoch 3/30
2560/2560 [==============================] - 1s 234us/step - loss: 0.1603 - accuracy: 0.9316 - true_positive: 597.8250 - val_loss: 0.1720 - val_accuracy: 0.9281 - val_true_positive: 142.0500
Epoch 4/30
2560/2560 [==============================] - 1s 235us/step - loss: 0.1328 - accuracy: 0.9465 - true_positive: 624.5875 - val_loss: 0.1552 - val_accuracy: 0.9359 - val_true_positive: 143.2500
Epoch 5/30
2560/2560 [==============================] - 1s 236us/step - loss: 0.1249 - accuracy: 0.9527 - true_positive: 598.5125 - val_loss: 0.1366 - val_accuracy: 0.9500 - val_true_positive: 150.2000
Epoch 6/30
2560/2560 [==============================] - 1s 240us/step - loss: 0.1092 - accuracy: 0.9625 - true_positive: 624.6750 - val_loss: 0.1045 - val_accuracy: 0.9578 - val_true_positive: 153.0000
Epoch 7/30
2560/2560 [==============================] - 1s 238us/step - loss: 0.1035 - accuracy: 0.9594 - true_positive: 631.0625 - val_loss: 0.1217 - val_accuracy: 0.9625 - val_true_positive: 154.5500
Epoch 8/30
2560/2560 [==============================] - 1s 238us/step - loss: 0.1052 - accuracy: 0.9609 - true_positive: 622.9250 - val_loss: 0.0887 - val_accuracy: 0.9688 - val_true_positive: 156.5500
Epoch 9/30
2560/2560 [==============================] - 1s 233us/step - loss: 0.0759 - accuracy: 0.9734 - true_positive: 625.4500 - val_loss: 0.0769 - val_accuracy: 0.9688 - val_true_positive: 155.9000
Epoch 10/30
2560/2560 [==============================] - 1s 238us/step - loss: 0.0877 - accuracy: 0.9656 - true_positive: 618.9250 - val_loss: 0.0675 - val_accuracy: 0.9734 - val_true_positive: 159.4500
Epoch 11/30
2560/2560 [==============================] - 1s 235us/step - loss: 0.0582 - accuracy: 0.9812 - true_positive: 630.1250 - val_loss: 0.0836 - val_accuracy: 0.9734 - val_true_positive: 158.7500
Epoch 12/30
2560/2560 [==============================] - 1s 238us/step - loss: 0.0632 - accuracy: 0.9754 - true_positive: 626.4500 - val_loss: 0.0956 - val_accuracy: 0.9625 - val_true_positive: 153.0000
Epoch 13/30
2560/2560 [==============================] - 1s 233us/step - loss: 0.0747 - accuracy: 0.9766 - true_positive: 628.1750 - val_loss: 0.0679 - val_accuracy: 0.9750 - val_true_positive: 158.9500
Epoch 14/30
2560/2560 [==============================] - 1s 236us/step - loss: 0.0606 - accuracy: 0.9781 - true_positive: 642.2625 - val_loss: 0.0734 - val_accuracy: 0.9781 - val_true_positive: 159.0500
Epoch 15/30
2560/2560 [==============================] - 1s 238us/step - loss: 0.0592 - accuracy: 0.9793 - true_positive: 641.2625 - val_loss: 0.0606 - val_accuracy: 0.9766 - val_true_positive: 159.9000
Epoch 16/30
2560/2560 [==============================] - 1s 237us/step - loss: 0.0528 - accuracy: 0.9824 - true_positive: 634.0250 - val_loss: 0.0623 - val_accuracy: 0.9781 - val_true_positive: 159.7000
Epoch 17/30
2560/2560 [==============================] - 1s 240us/step - loss: 0.0411 - accuracy: 0.9840 - true_positive: 635.2250 - val_loss: 0.0614 - val_accuracy: 0.9781 - val_true_positive: 160.4500
Epoch 18/30
2560/2560 [==============================] - 1s 233us/step - loss: 0.0568 - accuracy: 0.9816 - true_positive: 640.4500 - val_loss: 0.1096 - val_accuracy: 0.9609 - val_true_positive: 154.5500
Epoch 19/30
2560/2560 [==============================] - 1s 236us/step - loss: 0.0594 - accuracy: 0.9812 - true_positive: 635.7000 - val_loss: 0.0640 - val_accuracy: 0.9812 - val_true_positive: 161.4500
Epoch 20/30
2560/2560 [==============================] - 1s 239us/step - loss: 0.0429 - accuracy: 0.9855 - true_positive: 634.3000 - val_loss: 0.0591 - val_accuracy: 0.9812 - val_true_positive: 160.9000
Epoch 21/30
2560/2560 [==============================] - 1s 235us/step - loss: 0.0535 - accuracy: 0.9812 - true_positive: 635.9500 - val_loss: 0.0584 - val_accuracy: 0.9781 - val_true_positive: 158.9500
Epoch 22/30
2560/2560 [==============================] - 1s 234us/step - loss: 0.0503 - accuracy: 0.9855 - true_positive: 648.6750 - val_loss: 0.0486 - val_accuracy: 0.9844 - val_true_positive: 161.5000
Epoch 23/30
2560/2560 [==============================] - 1s 235us/step - loss: 0.0434 - accuracy: 0.9855 - true_positive: 641.9000 - val_loss: 0.0535 - val_accuracy: 0.9750 - val_true_positive: 158.9500
Epoch 24/30
2560/2560 [==============================] - 1s 239us/step - loss: 0.0410 - accuracy: 0.9859 - true_positive: 630.1375 - val_loss: 0.0596 - val_accuracy: 0.9750 - val_true_positive: 159.5000
Epoch 25/30
2560/2560 [==============================] - 1s 237us/step - loss: 0.0285 - accuracy: 0.9891 - true_positive: 633.1375 - val_loss: 0.0572 - val_accuracy: 0.9797 - val_true_positive: 161.3500
Epoch 26/30
2560/2560 [==============================] - 1s 238us/step - loss: 0.0356 - accuracy: 0.9867 - true_positive: 642.5625 - val_loss: 0.0783 - val_accuracy: 0.9797 - val_true_positive: 160.9000
Epoch 27/30
2560/2560 [==============================] - 1s 238us/step - loss: 0.0521 - accuracy: 0.9832 - true_positive: 643.1125 - val_loss: 0.0584 - val_accuracy: 0.9828 - val_true_positive: 161.1500
Epoch 28/30
2560/2560 [==============================] - 1s 234us/step - loss: 0.0303 - accuracy: 0.9914 - true_positive: 646.0875 - val_loss: 0.0505 - val_accuracy: 0.9812 - val_true_positive: 160.5000
Epoch 29/30
2560/2560 [==============================] - 1s 238us/step - loss: 0.0420 - accuracy: 0.9844 - true_positive: 630.9375 - val_loss: 0.0687 - val_accuracy: 0.9766 - val_true_positive: 158.6000
Epoch 30/30
2560/2560 [==============================] - 1s 237us/step - loss: 0.0278 - accuracy: 0.9922 - true_positive: 653.9000 - val_loss: 0.0566 - val_accuracy: 0.9828 - val_true_positive: 160.3500

True positives: 320 / 331
True negatives: 309 / 309
False positives: 0 / 309
False negatives: 11 / 331

Accuracy: 0.9828125
Recall: 0.9667673716012085
Precision: 1.0
F1: 0.9831029185867896
```