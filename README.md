# ESMSec

1.You should find the corresponding data at https://bmbl.bmi.osumc.edu/HBFP/, Or use the data we provide

2.Unzip and modify the path to the file in the code

3.run main.py

Reminder: In the data file, there are three processed data sets of human body fluids. The file with the word "last" added before the name of body fluids is the training data when predicting secreted proteins of each body fluid. In predicting secreted proteins, the training set of the training model and the test set are combined into a new training set for the training model, and the previous test set is used to adjust the parameters. Make predictions on data sets with the word "eval"
