import numpy as np

from Simulation_DAN import *
from Network_DAN import *
import torch
import vessl
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

if __name__=="__main__":
    problem_dir='/output/problem_set/'
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    model_dir='/output/model/ppo/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    history_dir='/output/history/'
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

                
    history=pd.DataFrame(history)
    validation_history=pd.DataFrame(validation_history)
    history.to_excel(history_dir+'history.xlsx', sheet_name='Sheet', index=False)
    validation_history.to_excel(history_dir + 'valid_history.xlsx', sheet_name='Sheet', index=False)

