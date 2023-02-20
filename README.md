# Wave2BoUNdary
Internship at CoML group. Unsupervised word segmentation in raw speech.



Run "python wave2vecffnn.py" to train the model. Please select inside the file the training settings, language and pretrained model.
This command will create a folder with the execution date. Such code will be used to identify the trained model.
Training curves and model weights will be saved inside this folder.


Run "python predict_zero.py" to predict from a trained model. Please select inside the folder the language and the identifying code for the desired model.
Predictions will be saved in the "dummy.csv" file. Also, TDE will be executed and the results will be printed in the console.
