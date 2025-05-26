import torch.distributed as dist
import torch
import torch.nn as nn
import torch.optim

import math
from functools import partial

from ablang_train.kfac_utils import KFACLinear


def save_input(mod, inp, kfac_ema, beta1, device, dtype):
    if mod.training:
        with torch.no_grad():
            mid = id(mod)
            A = inp[0].detach()
            A = A.view(-1, A.size(-1))
            
            if mod.bias is not None:
                ones = torch.ones(A.size(0), 1, dtype=A.dtype, device=A.device)
                A = torch.cat([A, ones], dim=1)
            A_factor = A.t() @ A / A.size(0)
            A_factor = A_factor.to(device=device, dtype=dtype)
            
            #print(f"[{A.device}, {mod.training}] A shape: {A.shape}")
            kfac_ema[mid]['AA'].mul_(beta1).add_(A_factor, alpha=1 - beta1)
            del A, A_factor
            if mod.bias is not None:
                del ones

def save_grad(mod, grad_input, grad_output, kfac_ema, beta2, device, dtype):
    if mod.training:
        with torch.no_grad():
            mid = id(mod)
            G = grad_output[0].detach()
            G = G.view(-1, G.size(-1))
            G_factor = G.t() @ G / G.size(0)
            G_factor = G_factor.to(device=device, dtype=dtype)
            
            #print(f"[{G.device}, {mod.training}] G shape: {G.shape}")
            kfac_ema[mid]['GG'].mul_(beta2).add_(G_factor, alpha=1 - beta2)
            del G, G_factor

            return grad_input

class KOAP(torch.optim.Optimizer):
    def __init__(self, params, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, \
                 damping=0.01, precondition_warmup: int = 10,\
                 precondition_frequency: int = 100, top: float = 1.0, mode: str = 'eigh',
                 # KFAC betas are now part of KFACLinear, dict_device/dtype removed
                 # rank and world_size might still be needed for dist.all_reduce if not handled by PL automatically
                 # For now, assuming dist.is_initialized() and dist.get_rank()/world_size() can be used if needed
                 # Or, these can be passed if PL doesn't set them up in a way accessible here.
                 # The example optimizer seems to get them from os.environ, which is one way.
                 # Let's assume for now that all_reduce will work correctly in a DDP context.
                 ):
        defaults = dict(lr=lr, betas=betas, eps=eps, damping=damping,\
                        precondition_warmup=precondition_warmup,\
                        precondition_frequency=precondition_frequency,\
                        top=top, mode=mode)
        super().__init__(params, defaults)

        self._step = 0
        # top and mode are now in defaults, can be accessed via group['top'] or stored directly if preferred
        self._top = top 
        self._mode = mode 

        self.linear_modules = []
        self.param_to_module = {}
        param_ids = {id(p) for group in self.param_groups for p in group['params']}

        # Iterate through the model's modules to find KFACLinear layers associated with the optimizer's params
        # print('MODULE LIST')
        for module in model.modules():
            # print(module)
            if isinstance(module, KFACLinear): # Specifically look for KFACLinear
                # Check if any of this module's parameters are in the optimizer's param groups
                module_param_ids = {id(p) for p in module.parameters()}
                if not param_ids.isdisjoint(module_param_ids):
                    self.linear_modules.append(module)
                    for p_id in module_param_ids:
                        if p_id in param_ids:
                             self.param_to_module[p_id] = module
        
        # Verify all params given to optimizer belong to a tracked KFACLinear module
        # This check is important because KOAP is designed for KFACLinear params.
        # If other params are passed, they won't be processed correctly.
        all_optim_param_ids = {id(p) for group in self.param_groups for p in group['params']}
        tracked_param_ids = set(self.param_to_module.keys())
        untracked = all_optim_param_ids - tracked_param_ids
        if untracked:
            # This means some parameters passed to KOAP do not belong to KFACLinear layers.
            # This indicates a misconfiguration in how parameters are split between optimizers.
            # The user's request was to use KOAP for KFACLinear and AdamW for others.
            # So, KOAP should only receive KFACLinear parameters.
            # For now, we raise an error. If the intention is to mix, KOAP would need modification.
            raise ValueError(f"KOAP Optimizer was given parameters that do not belong to KFACLinear modules. Untracked param IDs: {untracked}")

        self.kfac_q = {id(m): {'AA_eigvecs': None, 'GG_eigvecs': None} for m in self.linear_modules}
        # Removed self.handles and hook registration/removal, as EMAs are in KFACLinear

    def update_kfac_q(self):
        eps = self.defaults['damping'] # Use damping from defaults

        for module in self.linear_modules:
            mid = id(module)

            # Ensure ema_AA and ema_GG are present, as per KFACLinear new design
            if not hasattr(module, 'ema_AA') or not hasattr(module, 'ema_GG'):
                raise AttributeError(f"Module {module} is expected to have 'ema_AA' and 'ema_GG' attributes.")

            AA = module.ema_AA.data.detach() # Get from KFACLinear layer
            GG = module.ema_GG.data.detach() # Get from KFACLinear layer

            # if dist.is_initialized():
            #     dist.all_reduce(AA, op=dist.ReduceOp.AVG)
            #     dist.all_reduce(GG, op=dist.ReduceOp.AVG)
            
            # Regularization term based on example
            # Ensure AA and GG are not empty and have diag method
            if AA.numel() == 0 or GG.numel() == 0:
                # This can happen if a KFACLinear layer has 0 input/output features, or EMAs are not updated
                # Skip eigen-decomposition for such layers if they occur
                print(f"Warning: ema_AA or ema_GG for module {mid} is empty. Skipping eigen-decomposition.")
                self.kfac_q[mid]['AA_eigvecs'] = None
                self.kfac_q[mid]['GG_eigvecs'] = None
                continue
                
            # print(f'AA {AA.diag()[:20]}')
            lambda_reg_numerator = torch.outer(AA.diag(), GG.diag()).mean().item()
            if lambda_reg_numerator < 0:
                 # this can happen if diagonals have negative values due to numerical instability or init
                 # using abs to prevent sqrt of negative. Or use a small floor like 1e-10.
                 lambda_reg_numerator = abs(lambda_reg_numerator)
            # print(lambda_reg_numerator, eps)
            lambda_reg = math.sqrt(lambda_reg_numerator * eps)

            current_mode = self.defaults['mode'] # Use mode from defaults

            # Conditional CPU fallback for eigh on MPS
            device_AA = AA.device
            device_GG = GG.device

            if self.kfac_q[mid]['AA_eigvecs'] is None or current_mode == 'eigh':
                if AA.numel() > 0:
                    eye_AA = torch.eye(AA.size(0), dtype=AA.dtype, device=device_AA)
                    if device_AA.type == 'mps':
                        _, AA_qr_cpu = torch.linalg.eigh(AA.cpu() + lambda_reg * eye_AA.cpu())
                        AA_qr = AA_qr_cpu.to(device_AA)
                    else:
                        # print(AA.shape, lambda_reg, eye_AA.shape)
                        _, AA_qr = torch.linalg.eigh(AA + lambda_reg * eye_AA)
                else: AA_qr = torch.empty(0,0, dtype=AA.dtype, device=device_AA)
                
                if GG.numel() > 0:
                    eye_GG = torch.eye(GG.size(0), dtype=GG.dtype, device=device_GG)
                    if device_GG.type == 'mps':
                        _, GG_qr_cpu = torch.linalg.eigh(GG.cpu() + lambda_reg * eye_GG.cpu())
                        GG_qr = GG_qr_cpu.to(device_GG)
                    else:
                        _, GG_qr = torch.linalg.eigh(GG + lambda_reg * eye_GG)
                else: GG_qr = torch.empty(0,0, dtype=GG.dtype, device=device_GG)
            
            elif current_mode == 'power': 
                AA_qr_prev = self.kfac_q[mid]['AA_eigvecs']
                GG_qr_prev = self.kfac_q[mid]['GG_eigvecs']
                
                # Power iteration should ideally also respect MPS device for intermediate products
                # Assuming AA and GG are already on the correct device (device_AA, device_GG)
                if AA.numel() > 0 and AA_qr_prev is not None and AA_qr_prev.numel() > 0:
                    AA_qr_new = AA @ AA_qr_prev
                    if device_AA.type == 'mps': # QR decomposition might also need fallback if problematic
                        AA_qr_cpu, _ = torch.linalg.qr(AA_qr_new.cpu())
                        AA_qr = AA_qr_cpu.to(device_AA)
                    else:
                        AA_qr, _ = torch.linalg.qr(AA_qr_new)
                else: AA_qr = torch.empty(0,0, dtype=AA.dtype, device=device_AA)

                if GG.numel() > 0 and GG_qr_prev is not None and GG_qr_prev.numel() > 0:
                    GG_qr_new = GG @ GG_qr_prev
                    if device_GG.type == 'mps': # QR decomposition might also need fallback
                        GG_qr_cpu, _ = torch.linalg.qr(GG_qr_new.cpu())
                        GG_qr = GG_qr_cpu.to(device_GG)
                    else:
                        GG_qr, _ = torch.linalg.qr(GG_qr_new)
                else: GG_qr = torch.empty(0,0, dtype=GG.dtype, device=device_GG)
            else:
                raise ValueError(f"Unknown KFAC Q update mode: {current_mode}")

            top_k_ratio = self.defaults['top'] # Use top from defaults
            if top_k_ratio < 1.0:
                if AA_qr.numel() > 0 :
                    k_AA = math.ceil(AA_qr.size(1) * top_k_ratio)
                    AA_qr = AA_qr[:, :k_AA]
                if GG_qr.numel() > 0:
                    k_GG = math.ceil(GG_qr.size(1) * top_k_ratio)
                    GG_qr = GG_qr[:, :k_GG]
            
            self.kfac_q[mid]['AA_eigvecs'] = AA_qr
            self.kfac_q[mid]['GG_eigvecs'] = GG_qr

    def project(self, grad, kfac_q_module, has_bias=False, is_bias_grad=False):
        AA_eigvecs = kfac_q_module['AA_eigvecs']
        GG_eigvecs = kfac_q_module['GG_eigvecs']

        if AA_eigvecs is None or GG_eigvecs is None or AA_eigvecs.numel() == 0 or GG_eigvecs.numel() == 0:
            # If no eigenvectors (e.g. empty EMA, or skipped update), return grad as is
            return grad

        if not is_bias_grad:
            # Weight gradient projection
            # grad: (out_features, in_features)
            # GG_eigvecs: (out_features, k_GG)
            # AA_eigvecs: (in_features_aug, k_AA) -> (in_features, k_AA) if bias
            current_AA_eigvecs = AA_eigvecs
            if has_bias:
                # KFACLinear stores ema_AA for augmented input (features + bias term)
                # For weight grad, we need AA for non-augmented input features.
                if AA_eigvecs.size(0) == grad.size(1) + 1: # sanity check size
                    current_AA_eigvecs = AA_eigvecs[:-1, :] 
                # if AA_eigvecs.size(0) != grad.size(1): error or warning
            
            # Ensure dimensions match for matmul
            if GG_eigvecs.size(0) != grad.size(0) or current_AA_eigvecs.size(0) != grad.size(1):
                 # This would be a mismatch, return grad to avoid crashing
                 # print(f"Warning: Mismatch in project dims. Grad: {grad.shape}, GG: {GG_eigvecs.shape}, AA: {current_AA_eigvecs.shape}")
                 return grad

            return GG_eigvecs.T @ grad @ current_AA_eigvecs
        else:
            # Bias gradient projection
            # grad: (out_features,)
            # GG_eigvecs: (out_features, k_GG)
            # Bias is like a weight connected to a constant 1 input.
            # So, it's mainly affected by GG (output grads covariance)
            # The example project: return GG_bias.T @ grad
            if GG_eigvecs.size(0) != grad.size(0):
                # print(f"Warning: Mismatch in project_bias dims. Grad: {grad.shape}, GG: {GG_eigvecs.shape}")
                return grad
            return GG_eigvecs.T @ grad.unsqueeze(-1) # grad needs to be [out_features, 1]

    def project_back(self, grad_projected, kfac_q_module, has_bias=False, is_bias_grad=False):
        AA_eigvecs = kfac_q_module['AA_eigvecs']
        GG_eigvecs = kfac_q_module['GG_eigvecs']
        
        if AA_eigvecs is None or GG_eigvecs is None or AA_eigvecs.numel() == 0 or GG_eigvecs.numel() == 0:
            return grad_projected
        
        if not is_bias_grad:
            AA_eigvecs_for_transform = AA_eigvecs
            if has_bias: 
                # If layer has bias, AA_eigvecs from kfac_q is for augmented space (in_features + 1).
                # For weight projection, we need the part corresponding to in_features.
                if AA_eigvecs.size(0) > 0: # Ensure not empty before slicing
                    AA_eigvecs_for_transform = AA_eigvecs[:-1, :] # Shape: (in_features, k_dim)
                # If AA_eigvecs was empty, AA_eigvecs_for_transform remains empty, handled by checks below.

            # grad_projected is (k_GG, k_AA_effective)
            # GG_eigvecs is (out_features, k_GG)
            # AA_eigvecs_for_transform.T is (k_AA_effective, in_features)
            
            if GG_eigvecs.numel() == 0 or grad_projected.numel() == 0 or AA_eigvecs_for_transform.numel() == 0:
                return grad_projected 

            # Dimension checks for matrix multiplication compatibility
            if GG_eigvecs.size(1) != grad_projected.size(0) or \
               grad_projected.size(1) != AA_eigvecs_for_transform.size(1): 
                # print(f"Warning: Mismatch in project_back weight dims. GradP: {grad_projected.shape}, GG: {GG_eigvecs.shape}, AA_transform: {AA_eigvecs_for_transform.shape}")
                return grad_projected

            return GG_eigvecs @ grad_projected @ AA_eigvecs_for_transform.T
        else:
            # grad_projected: (k_GG, 1) from project()
            # GG_eigvecs: (out_features, k_GG)
            if GG_eigvecs.size(1) != grad_projected.size(0):
                # print(f"Warning: Mismatch in project_back_bias dims. GradP: {grad_projected.shape}, GG: {GG_eigvecs.shape}")
                return grad_projected.squeeze(-1) if grad_projected.ndim > 1 else grad_projected
            
            res = GG_eigvecs @ grad_projected
            return res.squeeze(-1) # Ensure output is (out_features,)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        group = self.param_groups[0] # Assuming one param group for KOAP, as in example.
                                      # If multiple groups, would need to iterate or ensure defaults are consistent.

        if (self._step % group['precondition_frequency'] == 0) or \
           (self._step < group['precondition_warmup']):
            # print("Updating preconditioner...")
            self.update_kfac_q()
            # print("Updating preconditioner... done")
        # else:
            # print("skipping precondition update")
        self._step += 1
        
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data # Original gradient
            state = self.state[p]

            module = self.param_to_module[id(p)]
            mid = id(module)
            kfac_q_module = self.kfac_q[mid]
            has_bias = module.bias is not None
            is_bias_grad = (p is module.bias)

            # Project original gradient for exp_avg_sq accumulation
            # The example Adam uses grad_projected.square() for exp_avg_sq
            # This means we need to project the *original* full gradient, not the exp_avg
            grad_projected_for_sq = self.project(grad, kfac_q_module, has_bias=has_bias, is_bias_grad=is_bias_grad)

            if len(state) == 0:
                state['step'] = 0
                # exp_avg is for *original* gradients, as in Adam / example
                state['exp_avg'] = torch.zeros_like(p.data)
                # exp_avg_sq is for *projected* gradients, as in example
                state['exp_avg_sq'] = torch.zeros_like(grad_projected_for_sq) # Initialize with projected shape

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']
            eps = group['eps']

            state['step'] += 1

            # Update first moment (exp_avg) with original gradient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            
            # Update second moment (exp_avg_sq) with projected gradient squared
            exp_avg_sq.mul_(beta2).addcmul_(grad_projected_for_sq, grad_projected_for_sq, value=1 - beta2)

            # Denominator for Adam update using projected space
            # denom = exp_avg_sq.sqrt().add_(eps)
            # Bias correction for Adam
            bias_correction1 = 1.0 - beta1 ** state['step']
            bias_correction2 = 1.0 - beta2 ** state['step']
            
            # Per example: step_size = group["lr"] * (bias_correction2 ** .5) / bias_correction1
            # This step_size is applied to (project_back( project(exp_avg) / denom ))
            # So, project exp_avg first
            exp_avg_projected = self.project(exp_avg, kfac_q_module, has_bias=has_bias, is_bias_grad=is_bias_grad)
            
            # Denominator for Adam in projected space
            # Need to handle shape if exp_avg_sq was for bias (1D) and exp_avg_projected is (k_G, 1)
            if is_bias_grad and exp_avg_sq.ndim == 1 and exp_avg_projected.ndim == 2:
                # exp_avg_sq is (k_G), exp_avg_projected is (k_G, 1)
                denom = exp_avg_sq.sqrt().add_(eps).unsqueeze(-1) # make it (k_G, 1)
            else:
                denom = exp_avg_sq.sqrt().add_(eps)

            # Adam update in projected space
            update_projected = exp_avg_projected / denom
            
            # Project back to original parameter space
            norm_grad = self.project_back(update_projected, kfac_q_module, has_bias=has_bias, is_bias_grad=is_bias_grad)
            
            step_size = group["lr"] * (math.sqrt(bias_correction2) / bias_correction1)
            
            p.data.add_(norm_grad, alpha=-step_size)

        return loss

