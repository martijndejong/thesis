# Pilot Skill Level Identification Using Convolutional Neural Networks

## Overview
This project introduces a novel deep learning approach to evaluate human controller (HC) skill levels in manual control systems. Unlike traditional methods that rely on long-duration, quasi-linear models, our technique utilizes short windows of raw HC time signals, capturing both linear and non-linear behaviors for a more comprehensive assessment.

A schematic description of the HC skill level classification task is displayed below:
<img src="https://github.com/martijndejong/thesis/assets/12080489/6a9b34e3-6418-42a2-99ee-83d909f73560" alt="Pilot Classification Scheme" width="50%"/>

## Abstract

Mathematical human controller (HC) models are widely used in tuning manual control systems and for understanding human performance. Typically, quasi-linear HC models are used, which can accurately capture the linear portion of HCs’ behavior, averaged over a long measurement window. This paper presents a deep learning HC skill-level evaluation method that works on short windows of raw HC time signals, and accounts for both the linear and non-linear portions of HC behavior. This deep learning approach is applied to data from a previous skill training experiment performed in the SIMONA Research Simulator at TU Delft. Additional human control data is generated using cybernetic HC model simulations. The results indicate that the deep learning evaluation method is successful in predicting HC skill level with 85-90% validation accuracy, but that training the classifier solely on simulated HC data reduces this accuracy by 15-25%. Inspection of the results especially shows a strong sensitivity of the classifier to the presence of remnant in the simulated training data. In conclusion, these results reveal that current quasi-linear HC model simulations, and in particular the remnant portion, do not adequately capture real time-domain HC behavior to allow effective training-data augmentation.

## Key Features
- **Deep Learning Model**: Utilizes deep convolutional neural networks (CNNs) to classify raw time series of manual control behavior as either 'skilled' or 'unskilled.'
- **Cybernetic Data Augmentation**: Employs cybernetic HC model simulations to generate additional human control data, enhancing the training process.
- **Explainable AI Techniques**: Utilizes SHapley Additive exPlanations (SHAP) and activation maximization for insightful model interpretation.
- **High Accuracy**: Achieves 85-90% validation accuracy in predicting HC skill levels, demonstrating the effectiveness of integrating deep learning with cybernetic models.

## Methodology
1. **Data Collection**: Utilized data from a skill training experiment conducted in the SIMONA Research Simulator at TU Delft, enriched with simulated human control data for training.
2. **Classification Rationale**: Treated HC skill level assessment as a Time Series Classification (TSC) problem, using Multivariate Time Series (MTS) of HC behavior.
3. **Model Training and Validation**: Trained deep CNNs with both real and simulated data, assessing the impact of cybernetic data augmentation on model performance.

## Results and Findings
- The integration of cybernetic data augmentation significantly contributes to the training dataset, albeit with a noted accuracy reduction when solely relying on simulated data.
- The classifier demonstrated sensitivity to the presence of 'remnant' in training data, underscoring the importance of capturing non-linear aspects of HC behavior.

## Conclusion
This work presents a promising step towards real-time, comprehensive evaluation of HC skill levels, leveraging the strengths of deep learning and cybernetic modeling. Future enhancements should focus on improving the simulation of non-linear HC behaviors to further augment data quality and classifier performance.

## Screenshots
<img src="https://github.com/martijndejong/thesis/assets/12080489/3b1b9bdf-cdab-4292-a0ff-b998c7af3138" alt="Screenshot of Results" width="100%"/>

<img src="https://github.com/martijndejong/thesis/assets/12080489/4b227adf-e6c6-4eb9-95b3-aa2c1ac67ebd" alt="Screenshot of Network Architecture" width="80%"/>

