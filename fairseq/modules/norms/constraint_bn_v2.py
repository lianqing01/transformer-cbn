from __future__ import division

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.autograd import Function

class LagrangianFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        # input shape: [1, C, 1, 1]
        # weight shape: [1, C, 1, 1]
        # output shape: [1, C, 1, 1]
        ctx.save_for_backward(input, weight)
        output = input * weight
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * weight
        if ctx.needs_input_grad[1]:
            # gradient ascent
            grad_weight = -1 * grad_output * input
        return grad_input, grad_weight


class Constraint_Norm(nn.Module):

    def __init__(self, num_features, weight_decay=1e-3,pre_affine=True, post_affine=True):
        super(Constraint_Norm, self).__init__()
        self.num_features = num_features
        self.pre_affine=pre_affine
        self.post_affine = post_affine
        self.set_dim()
        self.mu_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))
        self.gamma_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))

        #initialization
        self.mu_.data.fill_(0)
        self.gamma_.data.fill_(1)
        self.lagrangian = Constraint_Lagrangian(num_features,
                                                weight_decay=weight_decay)

        # strore mean and variance for reference
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.zeros(num_features))

        self.register_buffer("tracking_times", torch.tensor(0, dtype=torch.long))
        self.update_affine_only = False
        self.sample_noise=False
        self.noise_data_dependent=False
        self.sample_mean = None
        self.eps = 1e-4

    def store_norm_stat(self):
        self.noise_mu_.append(self.mu_.grad.clone().detach())
        self.noise_gamma_.append(self.gamma_.grad.clone().detach())

    def summarize_norm_stat(self):
        self.noise_mu_ = torch.stack(self.noise_mu_)
        self.noise_mu_sum = self.noise_mu_.clone()
        self.noise_mu_ = self.mu_.detach() - self.noise_mu_
        self.noise_gamma_ = torch.stack(self.noise_gamma_)
        self.nosie_gamma_sum = self.noise_gamma_.clone()
        self.noise_gamma_ = self.gamma_.detach() - self.noise_gamma_


        self.noise_mu_mean = torch.mean(self.noise_mu_, dim=0)
        self.noise_gamma_mean = torch.mean(self.noise_gamma_, dim=0)
        self.noise_mu_var = torch.var(self.noise_mu_, dim=0).clamp(min=0)
        self.noise_mu_std = torch.sqrt(self.noise_mu_var) * self.lambda_noise_weight
        #self.noise_gamma_var = torch.var(1 / (self.noise_gamma_**2+1e-5), dim=0).clamp(min=0)
        self.noise_gamma_var = torch.var(self.noise_gamma_**2, dim=0).clamp(min=0)
        self.noise_gamma_std = torch.sqrt(self.noise_gamma_var) * self.lambda_noise_weight



        self.noise_mu_ = []
        self.noise_gamma_ = []



    def get_mean_var(self):
        with torch.no_grad():
            mean = self.mean / (self.tracking_times + 1e-4)
            var = self.var / (self.tracking_times + 1e-4)
            mean = mean.abs().mean()
            var = var.abs().mean()
            var = self.var / (self.tracking_times + 1e-4)
            mean = mean.abs().mean()
            var = var.abs().mean()
        return mean, var



    def set_dim(self):
        self.feature_dim = [1, self.num_features, 1]
        self.norm_dim = [0, 2]
        if self.post_affine != False:
            self.post_affine_layer = Constraint_Affine2d(self.num_features)

    def _initialize_mu(self, with_affine=False):
        self.mean = self.mean / self.tracking_times
        self.old_mu_ = self.mu_.clone()

        self.mu_.data += self.mean.view(self.mu_.size())  * torch.sqrt(self.gamma_**2 + self.eps)

    def _initialize_gamma(self, with_affine=False):
        self.old_gamma_ = self.gamma_.clone()
        self.var = self.var / self.tracking_times
        self.var -= 1
        self.gamma_.data = torch.sqrt((self.var.view(self.gamma_.size())+1) * (self.gamma_**2+self.eps) ).data

    def _initialize_affine(self, resume=None):
        #temp = self.post_affine_layer.u_.data / (self.old_gamma_.data + self.eps)
        #self.post_affine_layer.u_.data.copy_(temp * self.gamma_.data))
        if resume is not None and resume is not False:

            self.post_affine_layer.u_.data = self.post_affine_layer.u_ * \
                            torch.sqrt(self.gamma_**2 + self.eps) / torch.sqrt(self.old_gamma_**2 + self.eps)

            self.post_affine_layer.c_.data = self.post_affine_layer.c_ + \
                        (self.post_affine_layer.u_ * (self.mu_ - self.old_mu_) / torch.sqrt(self.gamma_**2 + self.eps))


        #self.post_affine_layer.c_.data -= (temp -temp1)
        del self.old_mu_
        del self.old_gamma_


    def forward(self, x, pad_mask=None, is_encoder=False):
        '''
        input: T x B x C -> B x C x T
             : B x C x T -> T x B x C
        pad_mask: B x T (padding is True)
        '''

        # mean
        T, B, C = x.shape
        x = x.permute(1, 2, 0).contiguous()

        if self.training and self.sample_noise is True:
            temp_x = (x - self.mu_) / (torch.sqrt(self.gamma_**2 + self.eps))
            mean = self.lagrangian.get_weighted_mean(temp_x, self.norm_dim)
            var = self.lagrangian.get_weighted_var(temp_x, self.gamma_, self.norm_dim)
        # for mu
        if self.pre_affine:
            if self.sample_noise and self.training:
                    noise_mean = torch.normal(mean=self.sample_mean.fill_(1), std=self.sample_mean_std)
                    noise_mean = noise_mean.view(self.mu_.size()).clamp(min=0.1, max=10)
                    x = x - (self.mu_ * noise_mean.detach())
            else:
                x = x - self.mu_

        # for gamma
        if self.pre_affine:
            if self.sample_noise and self.training:
                    noise_var = torch.normal(mean=self.sample_mean.fill_(1), std=self.sample_var_std)
                    noise_var = noise_var.view(self.gamma_.size()).clamp(min=0.1, max=10)

                    x = x / torch.sqrt((self.gamma_ * noise_var.detach())**2 + self.eps)
            else:

                x = x / torch.sqrt(self.gamma_**2 + self.eps)
        if not self.training or self.sample_noise is False:
            mean = self.lagrangian.get_weighted_mean(x, self.norm_dim)
            var = self.lagrangian.get_weighted_var(x, self.gamma_, self.norm_dim)



        self.mean += mean.detach()
        self.var += var.detach()

        self.tracking_times += 1
        #self.summarize_x_hat.append(x.detach())
        if self.post_affine != False:
            x = self.post_affine_layer(x)
        x = x.permute(2, 0, 1).contiguous()
        return x



    def reset_norm_statistics(self):
        self.mean.fill_(0)
        self.var.fill_(0)
        self.tracking_times.fill_(0)



class Constraint_Norm1d(Constraint_Norm):
    def __init__(self, num_features, pre_affine=True, post_affine=True):
        super(Constraint_Norm1d, self).__init__(num_features, pre_affine=pre_affine, post_affine=post_affine)

    def set_dim(self):
        self.feature_dim = [1, self.num_features]
        self.norm_dim = [0]
        if self.post_affine != False:
            self.post_affine_layer = Constraint_Affine1d(self.num_features)

class Constraint_Norm2d(Constraint_Norm):
    def __init__(self, num_features, pre_affine=True, post_affine=True):
        super(Constraint_Norm2d, self).__init__(num_features, pre_affine=pre_affine, post_affine=post_affine)

    def set_dim(self):
        self.feature_dim = [1, self.num_features, 1, 1]
        self.norm_dim = [0, 2, 3]
        if self.post_affine != False:
            self.post_affine_layer = Constraint_Affine2d(self.num_features)




class Constraint_Lagrangian(nn.Module):

    def __init__(self, num_features, weight_decay=1e-4, lag_function=LagrangianFunction):
        super(Constraint_Lagrangian, self).__init__()
        self.num_features = num_features
        self.lambda_ = nn.Parameter(torch.Tensor(num_features))

        self.xi_ = nn.Parameter(torch.Tensor(num_features))
        self.lambda_.data.fill_(0)
        self.xi_.data.fill_(0)
        self.weight_decay = weight_decay
        self.lag_function = lag_function



    def get_weighted_mean(self, x, norm_dim):
        mean = x.mean(dim=norm_dim)
        self.weight_mean = LagrangianFunction.apply(mean, self.xi_)
        self.mean = mean
        return mean


    def get_weighted_var(self, x,  gamma, norm_dim):
        var = x**2 - 1
        var = var.mean(dim=norm_dim)
        self.weight_var = self.lag_function.apply(var, self.lambda_)
        self.var = var
        return var+1
    def get_weight_mean_var(self):
        return (self.weight_mean.mean(), self.weight_var.mean())

    def get_weight_mean_var_sum(self):
        return (self.weight_mean.sum(), self.weight_var.sum())



class Constraint_Affine(nn.Module):
    def __init__(self, num_features):
        super(Constraint_Affine, self).__init__()
        self.num_features = num_features
        self.set_dim()

        self.c_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))
        self.u_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))
        self.c_.data.fill_(0)
        self.u_.data.fill_(1)

    def set_dim(self):
        raise NotImplementedError


    def forward(self, x):
        return x * self.u_ + self.c_

class Constraint_Affine1d(Constraint_Affine):
    def __init__(self, num_features):
        super(Constraint_Affine1d, self).__init__(num_features)

    def set_dim(self):
        self.feature_dim = [1, self.num_features]


class Constraint_Affine2d(Constraint_Affine):
    def __init__(self, num_features):
        super(Constraint_Affine2d, self).__init__(num_features)

    def set_dim(self):
        self.feature_dim = [1, self.num_features, 1]


