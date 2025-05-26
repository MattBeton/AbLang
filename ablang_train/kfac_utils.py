import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KFACLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, ema_AA_buffer, ema_GG_buffer, beta_activations, beta_gradients, training):
        ctx.save_for_backward(input, weight, bias, ema_AA_buffer, ema_GG_buffer)
        ctx.betas = (beta_activations, beta_gradients)
        ctx.training = training
        
        if bias is not None:
            output = F.linear(input, weight, bias)
        else:
            output = F.linear(input, weight)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, ema_AA_buffer, ema_GG_buffer = ctx.saved_tensors
        beta_activations, beta_gradients = ctx.betas
        training = ctx.training

        grad_input = grad_weight = grad_bias = None

        if training:
            batch_dims = input.shape[:-1]
            in_features = input.shape[-1]
            out_features = grad_output.shape[-1]

            reshaped_input = input.reshape(-1, in_features)
            reshaped_grad_output = grad_output.reshape(-1, out_features)

            if bias is not None:
                ones = torch.ones(reshaped_input.size(0), 1, device=reshaped_input.device, dtype=reshaped_input.dtype)
                aug_input = torch.cat([reshaped_input, ones], dim=1)
            else:
                aug_input = reshaped_input
            
            num_examples = aug_input.size(0)

            # print(aug_input)
            # current_AA = (aug_input.t() @ aug_input) / num_examples
            # current_GG = (reshaped_grad_output.t() @ reshaped_grad_output) / num_examples
            




            # # DEBUG: Check for infinity/NaN in inputs
            # print(f"[DEBUG] Input shapes: aug_input={aug_input.shape}, grad_output={reshaped_grad_output.shape}")
            # print(f"[DEBUG] num_examples: {num_examples}")
            # print(f"[DEBUG] Device: {aug_input.device}")
            # print(f"[DEBUG] aug_input has inf: {torch.isinf(aug_input).any().item()}")
            # print(f"[DEBUG] aug_input has nan: {torch.isnan(aug_input).any().item()}")
            # print(f"[DEBUG] aug_input max: {aug_input.abs().max().item():.6e}")
            # print(f"[DEBUG] aug_input min: {aug_input.abs().min().item():.6e}")
            
            # Check the matrix multiplication result before division
            aa_matmul = aug_input.t() @ aug_input
            # print(f"[DEBUG] AA matmul has inf: {torch.isinf(aa_matmul).any().item()}")
            # print(f"[DEBUG] AA matmul has nan: {torch.isnan(aa_matmul).any().item()}")
            # print(f"[DEBUG] AA matmul max: {aa_matmul.abs().max().item():.6e}")
            # print(f"[DEBUG] AA matmul dtype: {aa_matmul.dtype}")
            
            # Check if division causes issues
            current_AA = aa_matmul / num_examples
            # print(f"[DEBUG] current_AA has inf: {torch.isinf(current_AA).any().item()}")
            # print(f"[DEBUG] current_AA has nan: {torch.isnan(current_AA).any().item()}")
            # print(f"[DEBUG] current_AA max: {current_AA.abs().max().item():.6e}")
            
            current_GG = (reshaped_grad_output.t() @ reshaped_grad_output) / num_examples
            
            # print(current_AA.diag()[:20])
            # print(current_AA.dtype) # This should now reflect kfac_matrix_dtype
            
            # # Check EMA buffer state before update
            # print(f"[DEBUG] ema_AA_buffer has inf before update: {torch.isinf(ema_AA_buffer).any().item()}")
            # print(f"[DEBUG] ema_AA_buffer max before update: {ema_AA_buffer.abs().max().item():.6e}")
            





            
            # print(f'current_AA {current_AA.diag()[:100]}')
            # print('beta_activattions: ', torch.isinf(beta_activations).any())
            # print('current_AA: ', torch.isinf(current_AA).any())
            ema_AA_buffer.mul_(beta_activations).add_(current_AA, alpha=1 - beta_activations)
            # print('buffer: ', torch.isinf(ema_AA_buffer).any())
            # print(ema_AA_buffer.flatten()[:100])
            ema_GG_buffer.mul_(beta_gradients).add_(current_GG, alpha=1 - beta_gradients)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight
        if ctx.needs_input_grad[1]:
            grad_w_temp = grad_output.reshape(-1, grad_output.size(-1)).t() @ input.reshape(-1, input.size(-1))
            grad_weight = grad_w_temp.view_as(weight)

        if bias is not None and ctx.needs_input_grad[2]:
            sum_dims = tuple(range(len(grad_output.shape) - 1))
            grad_bias = grad_output.sum(dim=sum_dims)
            grad_bias = grad_bias.view_as(bias)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class KFACLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, betas=(0.95, 0.95)):
        super(KFACLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.beta_activations, self.beta_gradients = betas

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            aug_in_features = in_features + 1
        else:
            self.register_parameter('bias', None)
            aug_in_features = in_features
        
        self.register_buffer('ema_AA', torch.zeros(aug_in_features, aug_in_features))
        self.register_buffer('ema_GG', torch.zeros(out_features, out_features))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight_to_use = self.weight.type_as(input)
        bias_to_use = self.bias.type_as(input) if self.bias is not None else None

        # print('input', torch.isinf(input).any())
        # print('ema_aa_pre_forward', torch.isinf(self.ema_AA).any())
        result = KFACLinearFunction.apply(input, weight_to_use, bias_to_use, 
                                        self.ema_AA, self.ema_GG, 
                                        self.beta_activations, self.beta_gradients, self.training)
        # print('ema_aa_post_forward', torch.isinf(self.ema_AA).any())
        return result
        
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, betas=({self.beta_activations},{self.beta_gradients})'


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


def _extract_patches(x, kernel_size, stride, padding):
    """
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def update_running_stat(aa, m_aa, stat_decay):
    # using inplace operation to save memory!
    m_aa *= stat_decay / (1 - stat_decay)
    m_aa += aa
    m_aa *= (1 - stat_decay)


class ComputeMatGrad:

    @classmethod
    def __call__(cls, input, grad_output, layer):
        if isinstance(layer, (nn.Linear, KFACLinear)):
            grad = cls.linear(input, grad_output, layer)
        elif isinstance(layer, nn.Conv2d):
            grad = cls.conv2d(input, grad_output, layer)
        else:
            raise NotImplementedError
        return grad

    @staticmethod
    def linear(input, grad_output, layer):
        """
        :param input: batch_size * input_dim
        :param grad_output: batch_size * output_dim
        :param layer: [nn.module] output_dim * input_dim
        :return: batch_size * output_dim * (input_dim + [1 if with bias])
        """
        with torch.no_grad():
            if layer.bias is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.unsqueeze(1)
            grad_output = grad_output.unsqueeze(2)
            grad = torch.bmm(grad_output, input)
        return grad

    @staticmethod
    def conv2d(input, grad_output, layer):
        """
        :param input: batch_size * in_c * in_h * in_w
        :param grad_output: batch_size * out_c * h * w
        :param layer: nn.module batch_size * out_c * (in_c*k_h*k_w + [1 if with bias])
        :return:
        """
        with torch.no_grad():
            input = _extract_patches(input, layer.kernel_size, layer.stride, layer.padding)
            input = input.view(-1, input.size(-1))  # b * hw * in_c*kh*kw
            grad_output = grad_output.transpose(1, 2).transpose(2, 3)
            grad_output = try_contiguous(grad_output).view(grad_output.size(0), -1, grad_output.size(-1))
            # b * hw * out_c
            if layer.bias is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.view(grad_output.size(0), -1, input.size(-1))  # b * hw * in_c*kh*kw
            grad = torch.einsum('abm,abn->amn', (grad_output, input))
        return grad


class ComputeCovA:

    @classmethod
    def compute_cov_a(cls, a, layer):
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, (nn.Linear, KFACLinear)):
            cov_a = cls.linear(a, layer)
        elif isinstance(layer, nn.Conv2d):
            cov_a = cls.conv2d(a, layer)
        else:
            # FIXME(CW): for extension to other layers.
            # raise NotImplementedError
            cov_a = None

        return cov_a

    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        # FIXME(CW): do we need to divide the output feature map's size?
        return a.t() @ (a / batch_size)

    @staticmethod
    def linear(a, layer):
        # a: batch_size * in_dim
        batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / batch_size)


class ComputeCovG:

    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            cov_g = cls.conv2d(g, layer, batch_averaged)
        elif isinstance(layer, (nn.Linear, KFACLinear)):
            cov_g = cls.linear(g, layer, batch_averaged)
        else:
            cov_g = None

        return cov_g

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        cov_g = g.t() @ (g / g.size(0))

        return cov_g

    @staticmethod
    def linear(g, layer, batch_averaged):
        # g: batch_size * out_dim
        batch_size = g.size(0)

        if batch_averaged:
            cov_g = g.t() @ (g * batch_size)
        else:
            cov_g = g.t() @ (g / batch_size)
        return cov_g


if __name__ == '__main__':
    def test_ComputeCovA():
        pass

    def test_ComputeCovG():
        pass






