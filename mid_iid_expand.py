import numpy as np
from funtest.test_pathlib import first_try
import sys
import os
sys.path.append('./util/')
from gen_expand import *
sys.path.append('./')
from policymodel import Policy,PolicyX

import argparse
parser=argparse.ArgumentParser(description='')
parser.add_argument('--fn',type=str,default='')
args = parser.parse_args()

fn_li=first_try(args.fn,'policy*.tar')
fn_li.sort(key=lambda x: os.stat(str(x)).st_mtime)
# fn_li=np.sort(fn_li)
print(fn_li)
# fn_li=np.hstack([fn_li[10:],fn_li[:10]])
# fn_li
i=0
##
# fn='./run/e/policy{}_inst_7757.pth.tar'.format(i)
# fn='./run/e/ea/policy{}_inst_7757.pth.tar'.format(i)
# fn='./run/i/ea/policy{}_inst_7757.pth.tar'.format(i)
class Nxt():
    def __init__(self):
        self.i=0
    def next(self):
        fn=str(fn_li[self.i])
        print(fn)
        m=Policy()
        self.log_prob,mid,iid=ck_parser(fn,m)
        self.m=m
        self.i=self.i+1
        return pd.DataFrame(np.vstack([iid[:-25],mid[:-25]]).T)
