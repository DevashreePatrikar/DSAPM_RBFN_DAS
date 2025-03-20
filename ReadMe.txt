1. Description 
This repository contains the Google Colab notebook and dataset used for our paper, "Dual-Stage Attention Mechanism for Robust Video Anomaly Detection and Localization," submitted to The Visual Computer. The project focuses on anomaly detection in video surveillance using CLSTM with Attention, RBFN optimization, and anomaly localization using a Spatial Pyramid Network with Attention. 

2. Dataset: UCSD Ped1, Ped2, Avenue, and ShanghaiTech

Link for datasets:

UCSD Ped1 and Ped2: http://svcl.ucsd.edu/projects/anomaly/dataset.htm
Avenue: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
ShanghaiTech: https://svip-lab.github.io/dataset/campus_dataset.html


3. Files Included

- ReadMe File
- Colab Notebooks: 

`[DSAPM_Train.ipynb]` → Contains the training DSAPM model. While training the model the batch size and number of epochs will vary for different datasets as given below:
*UCSD Ped1   : Epochs = 40;  Batch Size = 32*
*UCSD Ped2   : Epochs = 30;  Batch Size = 32*
*Avenue      : Epochs = 100; Batch Size = 64*
*ShanghaiTech: Epochs = 150; Batch Size = 128*
The epochs and batch size depend on the VRAM used. These values are for 16Gb of VRAM.
The training data consists of only normal events. 

`[Optimization_and_Thresholding.ipynb]` → Contains the optimization of frames scores using RBFN and thresholding. 

`[Test.ipynb]`→ The test data consisting of both normal and anomalous events is given to the model and evaluated for anomaly detection using the thresholding condition.

`[Evaluation_Metrics.ipynb]`→ Contains the code for obtaining the confusion matrix, ROC, AUC, Precision, Recall, F1-Score and Specificity.


`[DAS.ipynb]` → Contains the code to localize the anomaly. The DAS model is trained using the predicted images from DSAPM Model and a thresholding condition is decided. The test data is loaded and localized for anomaly.



3. How to Run
	a. Open the Colab notebook:  
   - As the file is shared via GitHub, open it directly in Colab.  

	b. Ensure all dependencies are installed by running:  
   ```python
   !pip install -r requirements.txt
   ```  
	c. Run the entire notebook step by step.  