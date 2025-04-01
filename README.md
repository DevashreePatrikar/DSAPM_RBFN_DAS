1. Description 
This repository contains the Google Colab notebook and dataset used for our paper, "Dual-Stage Attention Mechanism for Robust Video Anomaly Detection and Localization," currently submitted to a journal. The project focuses on anomaly detection in video surveillance using CLSTM with Attention, RBFN optimization, and anomaly localization using a Spatial Pyramid Network with Attention. 

2. Link for datasets:
UCSD Ped1 and Ped2: http://svcl.ucsd.edu/projects/anomaly/dataset.htm
Avenue: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
ShanghaiTech: https://svip-lab.github.io/dataset/campus_dataset.html

4. Files Included
- ReadMe File
- Colab Notebooks: 
	`[DSAPM_Train.ipynb]` → Contains the training DSAPM model. 
	`[Optimization_and_Thresholding.ipynb]` → Contains the optimization of frame scores using RBFN and thresholding. 
	`[Test.ipynb]`→ The test data, consisting of both normal and anomalous events, is given to the model and evaluated for anomaly detection using the thresholding condition.
	`[Evaluation_Metrics.ipynb]`→ Contains the code for obtaining the confusion matrix, ROC, AUC, Precision, Recall, F1-Score, and Specificity.
	`[DAS.ipynb]` → Contains the code to localize the anomaly. The DAS model is trained using the predicted images from the DSAPM Model, and a thresholding condition is decided. The test data is loaded and localized for anomaly.

4. Dependencies
-Ensure you have Python installed and install required packages using:
  -Python 3.x
  -TensorFlow
  -NumPy
  -Matplotlib
  -Seaborn
  -OpenCV
  -Scikit-learn
  -Pandas
  -SciPy

- For Google Colab users, dependencies are pre-installed, but you may need to install additional ones manually.
- Ensure that the runtime is set to GPU for better performance (Runtime → Change runtime type → GPU).

5. Prepare Datasets
- Download the datasets (UCSD Ped1, UCSD Ped2, Avenue, ShanghaiTech).
- Ensure they are stored in appropriate folders.
- Update file paths in the code if necessary.

6. How to Run
- Open the Colab notebook: As the file is shared via GitHub, open it directly in Colab.  
- Ensure all dependencies are installed as mentioned above #4
- Run the entire notebook step by step.  

7. Key Algorithms and Implementation Details
- The proposed model is divided into two components: Anomaly Detection and Anomaly Localization in surveillance videos. A common block used in both techniques is the DSAPM (Dual-Stage Attention Prediction Module), which consists of a Convolutional LSTM (CLSTM) with a Convolutional Block Attention Mechanism (CBAM). The DSAPM functions as a predictor, generating the next frame given a sequence of input frames. First, we train the DSAPM on a training dataset consisting of normal events. This allows the model to learn the expected patterns in normal video sequences, forming the basis for anomaly detection and localization. 

- Now that we have established the baseline, we move toward the implementation phase. The training process is detailed in `[DSAPM_Train.ipynb]`. We begin by preprocessing the data using a temporal shift mechanism, which prepares the model for learning future sequences. The DSAPM model architecture consists of 5 2D CLSTM blocks and 1 3D CLSTM block, with a CBAM block added after every 2D CLSTM block. The hyperparameters, including the number of epochs and batch size, vary based on the dataset: UCSD Ped1 (Epochs = 40, Batch Size = 32), UCSD Ped2 (Epochs = 30, Batch Size = 32), Avenue (Epochs = 100, Batch Size = 64), and ShanghaiTech (Epochs = 150, Batch Size = 128). These values are optimized for 16GB of VRAM, and extensive studies were conducted to determine them. The model is trained using Binary Cross-Entropy (BCE) loss and the Adam optimizer, where 19 frames are provided as input to predict the 20th frame. After training, we compute the Mean Squared Error (MSE) between the predicted and original frames, followed by the Peak Signal-to-Noise Ratio (PSNR) for further validation. Finally, the frame scores are calculated and saved in frame_scores_train.xlsx. The DSAPM model is saved as models/dsapm.h5.

- Refer file `[Optimization_and_Thresholding.ipynb]`. In this step, the frame scores are optimized using a Radial Basis Function Network (RBFN). First, we load the frame scores from 'frame_scores_train.xlsx' calculated in the previous section. We then apply the K-Means clustering algorithm and perform an RBF transformation using a Gaussian function to compute the optimized frame scores. The threshold is then determined as the maximum value of the optimized frame scores. These updated scores, along with the threshold, are saved in the same file. 

- Refer file `[Test.ipynb]`. The DSAPM model is tested on surveillance video frames to detect anomalies. The model, stored in dsapm.h5, is loaded and used for next-frame prediction. Test data is preprocessed and provided to the DSAPM Model to predict the next frame. Mean Squared Error (MSE) is computed between the predicted and original frame, followed by Peak Signal-to-Noise Ratio (PSNR) calculation. The PSNR values are normalized to obtain frame scores, which are saved in frame_scores_test.xlsx. The anomaly detection threshold, obtained from frame_scores_train.xlsx, is applied to classify frames as normal or anomalous, with results stored in the same file. This module detects frame-level anomaly.

- Refer file `[Evaluation_Metrics.ipynb]`. To evaluate the effectiveness of the proposed method, we compare the predicted results with the ground truth. First, we obtain the ground truth labels for the UCSD Ped1, UCSD Ped2, Avenue, and ShanghaiTech datasets, assigning 1 for anomaly and 0 for normal frames. For UCSD Ped1 and Ped2, the .mat files provide anomalous frame ranges, and the program marks frames within these ranges as 1. In the Avenue dataset, the Test.txt file specifies start and end indices of anomalies, and the program updates the corresponding frames to 1. For ShanghaiTech, the dataset provides per-frame masks (volLabel in .mat files), and the program checks each mask file—if it contains nonzero values, the corresponding frame is labeled as 1; otherwise, it remains 0. The final labels for each dataset are stored in gt_data_dict and saved in ground_truth_labels.xlsx. The predicted frame scores are obtained from frame_scores_test.xlsx and compared with the ground truth labels. Finally, we compute the confusion matrix, ROC curve, AUC, precision, recall, specificity, and F1-score to evaluate the model's performance.

- Refer file `[DAS.ipynb]` for the implementation of the DAS Model, which is used for anomaly localization in video surveillance by determining optical flow. The DAS model is built by integrating the Spatial Pyramid Network with CBAM, enhancing feature extraction. The DSAPM model (dsapm.h5) is then loaded to aid in training. Specifically, DSAPM first generates future frames from training sequences, and DAS learns to estimate optical flow from these predictions. The DAS model is trained using the Adam optimizer (learning rate = 0.0001) with MSE loss. The hyperparameters, including the number of epochs and batch size, vary based on the dataset: UCSD Ped1 (Epochs = 20, Batch Size = 16), UCSD Ped2 (Epochs = 15, Batch Size = 16), Avenue (Epochs = 50, Batch Size = 32), and ShanghaiTech (Epochs = 100, Batch Size = 64). These hyperparameters are optimized for 16GB VRAM, based on extensive empirical studies. Once trained, the End-Point Error (EPE) is computed between the predicted and ground truth optical flow, and a flow score is determined. The anomaly detection threshold is set based on this flow score. During testing, DAS predicts optical flow, and frames with flow scores exceeding the threshold are classified as anomalous. Anomalies are visualized using HSV-based optical flow representation for better interpretability. This module localizes the anomaly.

8. Citation
If you use this work, please cite:

**The details will be filled upon acceptance**
