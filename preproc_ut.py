#### import and reading data
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch
import gc
from funtest.test_pathlib import first_try
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(2018)
# %matplotlib inline
import sys
# sys.path.append('/opt/playground/web_traffic/')
# from util import threed_view
# from threed_view import *# from util import threed_view
sys.path.append('./')
from dst import *
from model import *
from policymodel import *
sys.path.append('./util')
from gen_expand import *
from threed_view import *
from meter import AverageMeter
import random
from preprocess import Preprocess
from preprocess import en_vec
torch.manual_seed(random.randint(1,2888))

# pre_processor=Preprocess('/mnt/osstb/tianchi/diaodu//')
pre_processor=Preprocess('/mnt/osstb/tianchi/diaodu///')
df_app_res_a,df_machine_a,df_ins_a,df_app_inter_a,df_ins_sum_a=pre_processor.run_pre()
del pre_processor
gc.collect()
pre_processor=Preprocess('/mnt/osstb/diaodu//')
df_app_res_b,df_machine_b,df_ins_b,df_app_inter_b,df_ins_sum_b=pre_processor.run_pre()
del pre_processor