# Pilot Skill Level Identification Using Convolutional Neural Networks

Researched and optimized artificial neural networks (machine learning) to identify pilot skill level in real-time


## Abstract

Mathematical human controller (HC) models are widely used in tuning manual control systems and for understanding human performance. Typically, quasi-linear HC models are used, which can accurately capture the linear portion of HCs’ behavior, averaged over a long measurement window. This paper presents a deep learning HC skill-level evaluation method that works on short windows of raw HC time signals, and accounts for both the linear and non-linear portions of HC behavior. This deep learning approach is applied to data from a previous skill training experiment performed in the SIMONA Research Simulator at TU Delft. Additional human control data is generated using cybernetic HC model simulations. The results indicate that the deep learning evaluation method is successful in predicting HC skill level with 85-90% validation accuracy, but that training the classifier solely on simulated HC data reduces this accuracy by 15-25%. Inspection of the results especially shows a strong sensitivity of the classifier to the presence of remnant in the simulated training data. In conclusion, these results reveal that current quasi-linear HC model simulations, and in particular the remnant portion, do not adequately capture real time-domain HC behavior to allow effective training-data augmentation.

![tempdelete](https://github.com/martijndejong/thesis/assets/12080489/6a9b34e3-6418-42a2-99ee-83d909f73560)
