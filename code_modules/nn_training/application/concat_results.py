import os

import joblib
import pandas as pd
from tqdm import tqdm

path = "code_modules/nn_training/application/results/"
file_list = os.listdir(path)

result = pd.DataFrame()
for file in tqdm(file_list):
    batch_i = int(file.split("_")[0].strip('batch'))
    minibatch_i = int(file.split("minibatch")[-1])
    result_i = pd.DataFrame(joblib.load(path + file),
                            columns=['minibatch_row', 'prob_false', 'prob_bac'])
    result_i['batch_i'] = batch_i
    result_i['minibatch_i'] = minibatch_i

    result = result.append(result_i, ignore_index=True)

result.memory_usage(deep=True).sum() / 1e6
result = result.query('prob_bac > 0.99')
