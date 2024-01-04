"""
**For most use cases, this can just be considered an internal class and
ignored.**

This module contains the abstract class AttackerStep as well as a few subclasses. 

AttackerStep is a generic way to implement optimizers specifically for use with
:class:`robustness.attacker.AttackerModel`. In general, except for when you want
to :ref:`create a custom optimization method <adding-custom-steps>`, you probably do not need to
import or edit this module and can just think of it as internal.
"""

import torch as ch

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True, 
                 min_value=0, max_value=1):
        '''
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
            min_value (float): the minimum value in the dataset (to clamp)
            max_value (float): the maximum value in the dataset (to clamp)
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad
        self.min_value = min_value
        self.max_value = max_value

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set

        Args:
            ch.tensor x : the input to project back into the feasible set.

        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).

        Parameters:
            g (ch.tensor): the raw gradient

        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x

### Instantiations of the AttackerStep class

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = ch.clamp(diff, -self.eps, self.eps)
        return ch.clamp(diff + self.orig_input, self.min_value, self.max_value)

    def step(self, x, g):
        """
        """
        step = ch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (ch.rand_like(x) - 0.5) * self.eps
        return ch.clamp(new_x, self.min_value, self.max_value)

# L2 threat model
class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return ch.clamp(self.orig_input + diff, self.min_value, self.max_value)

    def step(self, x, g):
        """
        """
        # Scale g so that each element of the batch is at least norm 1
        l = len(x.shape) - 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        """
        """
        new_x = x + (ch.rand_like(x) - 0.5).renorm(p=2, dim=0, maxnorm=self.eps)
        return ch.clamp(new_x, self.min_value, self.max_value)


# L2 step with norm enformed to be a particular length
class L2StepNormEnforced(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}

    But we enforce that the step is always at the edge of the L2 ball, rather than
    a max step. 
    """
    # The random initialization is correct, and was used for the random models.
    def project(self, x):
        """
        """
        raise NotImplementedError('Projection for L2StepNormEnforced not implemented')

    def step(self, x, g):
        """
        """
        raise NotImplementedError('Projection for L2StepNormEnforced not implemented')

    def random_perturb(self, x):
        """
        """
        # Random perturbation is debugged and correct (jfeather 5/3/2021)
        x_shape = x.shape
        perturbation = (ch.nn.functional.normalize(
                           ch.rand_like(x.view(x_shape[0], -1)) - 0.5, 
                           p=2,
                           dim=1) * self.eps).view(*x_shape)
        new_x = x + perturbation
        # Note: because there is a clamp the norm may be slightly different than 
        # self.eps
        return ch.clamp(new_x, self.min_value, self.max_value)


# Unconstrained threat model
class UnconstrainedStep(AttackerStep):
    """
    Unconstrained threat model, :math:`S = [0, 1]^n`.
    """
    def project(self, x):
        """
        """
        return ch.clamp(x, self.min_value, self.max_value)

    def step(self, x, g):
        """
        """
        return x + g * self.step_size

    def random_perturb(self, x):
        """
        """
        new_x = x + (ch.rand_like(x) - 0.5).renorm(p=2, dim=0, maxnorm=step_size)
        return ch.clamp(new_x, self.min_value, self.max_value)


# L-infinity threat model
class LinfCornersStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model, with the adversary
    always landing on the shell. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        raise NotImplementedError('Projection for LinfCornerStep not implemented')

    def step(self, x, g):
        """
        """
        raise NotImplementedError('Projection for LinfCornerStep not implemented')

    def random_perturb(self, x):
        """
        """
        # Random perturbation is debugged and correct (jfeather 5/3/2021)
        new_x = x + 2 * (ch.round(ch.rand_like(x)) - 0.5) * self.eps
        return ch.clamp(new_x, self.min_value, self.max_value)

