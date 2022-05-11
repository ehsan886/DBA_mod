import torch
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

AGGR_MEAN = 'mean'
AGGR_GEO_MED = 'geom_median'
AGGR_FOOLSGOLD='foolsgold'
AGGR_FLTRUST = 'fltrust'
AGGR_OURS = 'our_aggr'

ATTACK_DBA = 'dba'
ATTACK_TLF = 'targeted_label_flip'
MAX_UPDATE_NORM = 1000  # reject all updates larger than this amount
patience_iter=20

TYPE_LOAN='loan'
TYPE_CIFAR='cifar'
TYPE_MNIST='mnist'
TYPE_FMNIST='fmnist'
TYPE_TINYIMAGENET='tiny-imagenet-200'