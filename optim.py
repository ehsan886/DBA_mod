import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from torch import Tensor
from typing import List, Optional

from torch.optim.optimizer import Optimizer, required

from tensorflow import print as prnt


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        ref_params: List[Tensor],
        ref_grad_params: List[Tensor],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        inertia: float,
        minimizeDist: bool):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """
    #print("dummy", len(params), len(ref_params))

    if len(ref_params)!=0 and len(params)!=len(ref_params):
        #print(params, ref_params)
        print('params', params, '\n')
        print('ref_params', ref_params, '\n')
        sys.exit()

    if len(ref_grad_params)!=0 and len(params)!=len(ref_grad_params):
        print('params', params, '\n')
        print('ref_params', ref_grad_params, '\n')
        sys.exit()
    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        
        if ref_params is not None and len(ref_params) != 0 and minimizeDist:
            diff = 0
            ### if minimize euclidean distance
            #diff = param - ref_params[i]
            ### if minimize cosine similarity, ref_params contains the gradient of the reference network weights
            
            if ref_grad_params is not None:

                dir_sign = torch.sign(ref_params[i]-param)
                diff -= ref_params[i]-param
                diff -= ref_grad_params[i]
                diff = diff.mul_(0.5)
            
            #d_p = d_p.mul_(0)
            d_p = d_p.add(diff, alpha=inertia)
        
        #if diff_total==0:
            #print('diff_total 0')

        alpha = lr if maximize else -lr
        param.add_(d_p, alpha=alpha)




class SGD(Optimizer):
    def __init__(self, params, lr=required, ref_param_groups=None, ref_grad_param_groups=None, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, inertia=1.0, minimizeDist=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        #self.ref_param_groups = ref_param_groups
        super(SGD, self).__init__(params, defaults)
        #self.ref_param_groups = ref_param_groups
        self.inertia=inertia
        self.minimizeDist=minimizeDist

        if ref_param_groups is not None:
            self.__setrefparams__(ref_param_groups)
        else:
            self.ref_param_groups = ref_param_groups

        if ref_grad_param_groups is not None:
            self.__setrefgradparams__(ref_grad_param_groups)
        else:
            self.ref_grad_param_groups = ref_param_groups

    def filter_ref_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.
        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.
        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")
        '''
        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                name)
            else:
                param_group.setdefault(name, default)
        '''

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn("optimizer contains a parameter group with duplicate parameters; "
                          "in future, this will cause an error; "
                          "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        #self.ref_param_groups.append(param_group)
        return param_group
    
    def __setrefparams__(self, params):
        self.ref_param_groups = []

        param_groups = list(params)
        '''
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        '''
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.ref_param_groups.append(self.filter_ref_param_group(param_group))

    def __setrefgradparams__(self, params):
        self.ref_grad_param_groups = []

        param_groups = list(params)
        '''
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        '''
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.ref_grad_param_groups.append(self.filter_ref_param_group(param_group))

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
    
    

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        #for group in self.ref_param_groups:
            #for p in group['params']:

        if self.ref_param_groups is not None and len(self.param_groups)!=len(self.ref_param_groups):
            print(len(self.param_groups), len(self.ref_param_groups))
            sys.exit()

        if self.ref_grad_param_groups is not None and len(self.param_groups)!=len(self.ref_grad_param_groups):
            print(len(self.param_groups), len(self.ref_grad_param_groups))
            sys.exit()
        for i in range(len(self.param_groups)):
            group = self.param_groups[i]
            if self.ref_param_groups is not None:
                ref_group = self.ref_param_groups[i]
            if self.ref_grad_param_groups is not None:
                ref_grad_group = self.ref_grad_param_groups[i]
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
            ref_params=[]
            ref_grad_params=[]
            if self.ref_param_groups is not None:
                for p in ref_group['params']:
                    ref_params.append(p)
            if self.ref_grad_param_groups is not None:
                for p in ref_grad_group['params']:
                    ref_grad_params.append(p)
            
            sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  ref_params=ref_params,
                  ref_grad_params=ref_grad_params,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov,
                  maximize=maximize,
                  inertia=self.inertia,
                  minimizeDist=self.minimizeDist)
            

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

    ##  add add_param method

