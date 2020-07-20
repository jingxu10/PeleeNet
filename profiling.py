import torch
from torch.autograd import Variable
from torch.autograd import Function
import time

from peleenet import _DenseLayer, _DenseBlock, _StemBlock, BasicConv2d, PeleeNet

class Profiling(object):
    def __init__(self, model, pid, enabled=True):
        if isinstance(model, torch.nn.Module) is False:
            print("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self.pid = pid
        self.enabled = enabled
        self.record = {'forward':[], 'backward': []}
        self.profiling_on = True
        self.origin_call = {}
        self.hook_done = False
        self.layer_num = 0

    def __enter__(self):
        if self.enabled:
            self.start()

        return self

    def __exit__(self, *args):
        if self.enabled:
            self.stop()

    def __str__(self):
        ret = ""

        iter = len(self.record['forward']) / self.layer_num

        for i in range(iter):
            ret += "\n================================= Iteration {} =================================\n".format(i + 1)

            ret += "\nFORWARD TIME:\n\n"
            for j in xrange(self.layer_num):
                record_item = self.record['forward'][i * self.layer_num + j]
                ret += "layer{:3d}:          {:.6f} ms          ({})\n".format(j + 1, record_item[2] - record_item[1], record_item[0])

            ret += "\nBACKWARD TIME:\n\n"
            for j in (xrange(self.layer_num)):
                record_item = self.record['backward'][i * self.layer_num + self.layer_num - j - 1]
                try:
                    ret += "layer{:3d}:          {:.6f} ms          ({})\n".format(j + 1, record_item[2] - record_item[1], record_item[0])
                except:
                    # Oops, this layer doesn't execute backward post-hooks
                    pass

        return ret

    def start(self):
        if self.hook_done is False:
            self.hook_done = True
            self.hook_modules(self.model)

        self.profiling_on = True

        return self

    def stop(self):
        self.profiling_on = False

        return self

    def hook_modules(self, module, u_t_name=''):

        this_profiler = self

        sub_modules = module.__dict__['_modules']
        class_modules = [torch.nn.parallel.distributed.DistributedDataParallel, torch.nn.Sequential, _DenseLayer, _DenseBlock, _StemBlock, BasicConv2d, PeleeNet]
        class_containers = [torch.nn.modules.container.ModuleList]

        m_name = module._get_name()
        if u_t_name != '':
            m_name = u_t_name + '/' + m_name
        else:
            m_name = '[{}] {}'.format(self.pid, m_name)
        for name, sub_module in sub_modules.items():
            if any(isinstance(sub_module, c) for c in class_modules):
                self.hook_modules(sub_module, m_name)
            elif any(isinstance(sub_module, c) for c in class_containers):
                m_name = m_name + '/' + sub_module._get_name()
                for sub_sub_module in sub_module:
                    self.hook_modules(sub_sub_module, m_name)
            else:
                def forward_post_hook(module, input, output):
                    if (this_profiler.profiling_on):
                        msg = 'forward: {}/{}'.format(m_name, module._get_name())
                        print(msg)
                        _para = module.__dict__['_parameters']
                        for t in _para:
                            print('  [{}] {}: {} {} {}'.format(self.pid, t, _para[t].dtype, _para[t].shape, _para[t].requires_grad))
                        print('  [{}] input:'.format(self.pid))
                        id_t = 0
                        for t in input:
                            print('    [{}] {} {} {}'.format(self.pid, id_t, t.dtype, t.shape))
                            print('    [{}] {} {}'.format(self.pid, id_t, t))
                            id_t = id_t + 1
                        print('  [{}] output:'.format(self.pid))
                        id_t = 0
                        for t in input:
                            print('    [{}] {} {} {}'.format(self.pid, id_t, t.dtype, t.shape))
                            print('    [{}] {} {}'.format(self.pid, id_t, t))
                            id_t = id_t + 1

                def backward_post_hook(module, input, output):
                    if (this_profiler.profiling_on):
                        msg = 'backward: {}/{}'.format(m_name, module._get_name())
                        if hasattr(module, 'weight'):
                            msg = '{} {} {}'.format(msg, module.weight.dtype, module.weight.shape)
                        else:
                            msg = '{} {} {}'.format(msg, 'torch.None', 'torch.Size([])')
                        if hasattr(module, 'bias'):
                            msg = '{} {} {}'.format(msg, module.bias.dtype, module.bias.shape)
                        else:
                            msg = '{} {} {}'.format(msg, 'torch.None', 'torch.Size([])')
                        print(msg)
                        print('  [{}] input:'.format(self.pid))
                        id_t = 0
                        for t in input:
                            print('    [{}] {} {} {}'.format(self.pid, id_t, t.dtype, t.shape))
                            id_t = id_t + 1
                        print('  [{}] output:'.format(self.pid))
                        id_t = 0
                        for t in input:
                            print('    [{}] {} {} {}'.format(self.pid, id_t, t.dtype, t.shape))
                            id_t = id_t + 1

                sub_module.register_forward_hook(forward_post_hook)
                sub_module.register_backward_hook(backward_post_hook)
