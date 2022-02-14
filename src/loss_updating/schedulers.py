import torch
import math

class lr_lambda_class:
    def __init__(self, num_warmup_steps, num_training_steps):
        super().__init__()
        self.num_warmup_steps=num_warmup_steps
        self.num_training_steps = num_training_steps

    def __call__(self, current_step):          
        if current_step < self.num_warmup_steps: return float(current_step) / float(max(1, self.num_warmup_steps))

        return max(0.0, float(self.num_training_steps - current_step) / float(max(1, self.num_training_steps - self.num_warmup_steps)))

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """

    lr_lambda = lr_lambda_class(num_warmup_steps, num_training_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class lr_lambda_class_2:
    def __init__(self, num_warmup_steps, num_training_steps, num_cycles):
        super().__init__()
        self.num_warmup_steps=num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles

    def __call__(self, current_step):
        
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    lr_lambda = lr_lambda_class_2(num_warmup_steps, num_training_steps, num_cycles)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
