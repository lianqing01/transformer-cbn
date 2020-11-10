import torch
from fairseq.tasks.translation import TranslationTask
from . import register_task
from fairseq.modules.norms.constraint_bn_v2 import Constraint_Norm, Constraint_Lagrangian


@register_task('translation_cbn')
class TranslationCBNTask(TranslationTask):
    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):


        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0

        # cbn loss
        cbn_mean_loss = 0
        cbn_var_loss = 0
        cbn_mean = 0
        cbn_var = 0
        num_layer = 0
        for m in model.modules():
            if isinstance(m, Constraint_Lagrangian):
                num_layer += 1
                cbn_mean_loss += m.weight_mean.sum()
                cbn_var_loss += m.weight_var.sum()
                cbn_mean += m.mean.mean()
                cbn_var += m.var.mean()
        cbn_mean /= num_layer
        cbn_var /= num_layer
        cbn_mean_loss /= num_layer
        cbn_var_loss /= num_layer
        loss += self.args.cbn_loss_weight * (cbn_mean_loss + cbn_var_loss)

        optimizer.backward(loss)
        print("cbn_loss mean:{} var: {} cbn stat: mean: {} var: {}".format(cbn_mean_loss.item(), cbn_var_loss.item(),
                                                                           cbn_mean.item(), cbn_var.item()))
        return loss, sample_size, logging_output
