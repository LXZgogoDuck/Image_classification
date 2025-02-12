# python-project-classification
* implementations
1.training function:
First calculate the total loss and accurate data in training process. For accuracy, the correct data equals the number of train_outputs == target. Then the ratio equal to the number divide by the total number of training set i.e. len(training_loader.dataset)
Return training_acc and training_loss

2.test function:
“output” is what we get after putting data input into the model. After calculating all the test_loss, we can make a prediction to get the index of the max log-probability. Then calculate the correct testing data using “correct += prediction.eq(target.view_as(prediction)).sum().item()
“ Loss data and correct data divided by the total length of testing_set is the testing_loss and testing_acc.

3.plot function:
First set the value for x and y, x is given by integers in [0, epochs+1), and y is given by the arrays. (training_accuracies/training_loss etc.) 
Then set the x label and y label, plot the function and show the images.

4.random seed and multi-processing:
Firstly, edit the configuration file ‘minis.yaml’, add seed “-123 -321 -666”
In the main function, create three processes which can run together. Each process has a rank(label).
Pass the rank to the run function.                                i.e.p = mp.Process(target = run, args = (config, rank,))
In the run function, for each process with “label”, it has its own seed config.seed[rank], so we add different random seeds to different process.

5.plot mean function
Read each file with the recorded training and testing data, then do the plot.
