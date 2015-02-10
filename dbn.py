"""
The module for dbn construction.
"""

from sklearn.metrics import classification_report
import numpy as np
import argparse
from nolearn.dbn import DBN
from data_load import DataLoader
from model_evaluation import ResultEvaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help = 'Path to the train data')
    
    args = parser.parse_args()
    
    data = DataLoader(args.data_path)
    dbn = DBN(
              [data.trainX.shape[1], 500, data.classes_ammount],
              learn_rates = 0.2,
              learn_rate_decays = 0.9,
              epochs = 3,
              verbose = 1)
              
    print "Fitting model...."
    dbn.fit(data.trainX, data.trainY)
    
    result = dbn.predict_proba(data.testX)
    eval_data = ResultEvaluator(data)
    print "Loss of the model = " + str(eval_data.evaluate_loss(result))
    
    eval_data.display(dbn)
    
    preds = dbn.predict(data.testX)
    print classification_report(data.testY, preds, target_names = data.classes_names)

