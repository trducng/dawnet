# Implement the shake-shake algorithm
# @author: John Nguyen
# ==============================================================================
from torch.autograd import Function


class Shake(Function):
    """Implement the Shake regularization"""

    @staticmethod
    def forward(ctx, x1, x2, alpha, beta):
        """Perform the function forward pass"""
        ctx.save_for_backward(x1, x2, alpha, beta)
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        """Perform the autograd"""
        _, _, _, beta = ctx.saved_variables
        grad_x1 = grad_x2 = grad_alpha = grad_beta = None

        if ctx.needs_input_grad[0]:
            grad_x1 = grad_output * beta

        if ctx.needs_input_grad[1]:
            grad_x2 = grad_output * (1 - beta)

        return grad_x1, grad_x2, grad_alpha, grad_beta

shake = Shake.apply
