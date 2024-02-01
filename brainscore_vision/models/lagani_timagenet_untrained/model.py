
import functools

import torch.nn.functional as F
from brainscore_vision.model_helpers.check_submission import check_models
import torch
from torch import nn
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.

LAYERS = ['conv1', 'conv2', 'conv3', 'conv4', 'fc5']
BIBTEX = """custom model from https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb"""

NUM_CLASSES = 1000
img_size = 64
INPUT_SHAPE = (3, 64, 64)
MODEL_NAME = 'lagani-timagenet_untrained'

# UTILITY FUNCTIONS


def get_conv_output_shape(net):
    training = net.training
    net.eval()
    # In order to compute the shape of the output of the network convolutional layers, we can feed the network with
    # a simulated input and return the resulting output shape
    with torch.no_grad():
        res = tuple(
            net.get_conv_output(
                torch.ones(
                    1,
                    *
                    net.input_shape))[
                net.CONV_OUTPUT].size())[
                    1:]
    net.train(training)
    return res


def shape2size(shape):
    size = 1
    for s in shape:
        size *= s
    return size


# Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

# define your custom model here:
class Net(nn.Module):
    # Layer names
    CONV1 = 'conv1'
    POOL1 = 'pool1'
    BN1 = 'bn1'
    CONV2 = 'conv2'
    BN2 = 'bn2'
    CONV3 = 'conv3'
    POOL3 = 'pool3'
    BN3 = 'bn3'
    CONV4 = 'conv4'
    BN4 = 'bn4'
    # Symbolic name for the last convolutional layer providing extracted
    # features
    CONV_OUTPUT = BN4
    FC5 = 'fc5'
    BN5 = 'bn5'
    FC6 = 'fc6'
    CLASS_SCORES = FC6  # Symbolic name of the layer providing the class scores as output

    def __init__(self, input_shape=INPUT_SHAPE):
        super(Net, self).__init__()

        # Shape of the tensors that we expect to receive as input
        self.input_shape = input_shape

        # Here we define the layers of our network

        # First convolutional layer
        self.conv1 = HebbianMap2d(
            in_channels=3,
            # out_size=(8, 12),
            out_size=96,
            kernel_size=5,
            out=clp_cos_sim2d,
            eta=0.1,
        )  # 3 input channels, 8x12=96 output channels, 5x5 convolutions
        self.bn1 = nn.BatchNorm2d(96)  # Batch Norm layer

        # Second convolutional layer
        self.conv2 = HebbianMap2d(
            in_channels=96,
            # out_size=(8, 16),
            out_size=128,
            kernel_size=3,
            out=clp_cos_sim2d,
            eta=0.1,
        )  # 96 input channels, 8x16=128 output channels, 3x3 convolutions
        self.bn2 = nn.BatchNorm2d(128)  # Batch Norm layer

        # Third convolutional layer
        self.conv3 = HebbianMap2d(
            in_channels=128,
            # out_size=(12, 16),
            out_size=192,
            kernel_size=3,
            out=clp_cos_sim2d,
            eta=0.1,
        )  # 128 input channels, 12x16=192 output channels, 3x3 convolutions
        self.bn3 = nn.BatchNorm2d(192)  # Batch Norm layer

        # Fourth convolutional layer
        self.conv4 = HebbianMap2d(
            in_channels=192,
            # out_size=(16, 16),
            out_size=256,
            kernel_size=3,
            out=clp_cos_sim2d,
            eta=0.1,
        )  # 192 input channels, 16x16=256 output channels, 3x3 convolutions
        self.bn4 = nn.BatchNorm2d(256)  # Batch Norm layer

        self.conv_output_shape = get_conv_output_shape(self)

        # FC Layers (convolution with kernel size equal to the entire feature
        # map size is like a fc layer)

        self.fc5 = HebbianMap2d(
            in_channels=self.conv_output_shape[0],
            # out_size=(32, 32),
            out_size=1024,
            kernel_size=(
                self.conv_output_shape[1],
                self.conv_output_shape[2]),
            out=clp_cos_sim2d,
            eta=0.1,
        )  # conv_output_shape-shaped input, 15x20=300 output channels
        self.bn5 = nn.BatchNorm2d(1024)  # Batch Norm layer

        self.fc6 = HebbianMap2d(
            in_channels=1024,
            out_size=NUM_CLASSES,
            kernel_size=1,
            competitive=False,
            eta=0.1,
        )  # 300-dimensional input, 10-dimensional output (one per class)

    # This function forwards an input through the convolutional layers and
    # computes the resulting output
    def get_conv_output(self, x):
        # Layer 1: Convolutional + 2x2 Max Pooling + Batch Norm
        conv1_out = self.conv1(x)
        pool1_out = F.max_pool2d(conv1_out, 2)
        bn1_out = self.bn1(pool1_out)

        # Layer 2: Convolutional + Batch Norm
        conv2_out = self.conv2(bn1_out)
        bn2_out = self.bn2(conv2_out)

        # Layer 3: Convolutional + 2x2 Max Pooling + Batch Norm
        conv3_out = self.conv3(bn2_out)
        pool3_out = F.max_pool2d(conv3_out, 2)
        bn3_out = self.bn3(pool3_out)

        # Layer 4: Convolutional + Batch Norm
        conv4_out = self.conv4(bn3_out)
        bn4_out = self.bn4(conv4_out)

        # Build dictionary containing outputs of each layer
        conv_out = {
            self.CONV1: conv1_out,
            self.POOL1: pool1_out,
            self.BN1: bn1_out,
            self.CONV2: conv2_out,
            self.BN2: bn2_out,
            self.CONV3: conv3_out,
            self.POOL3: pool3_out,
            self.BN3: bn3_out,
            self.CONV4: conv4_out,
            self.BN4: bn4_out,
        }
        return conv_out

    # Here we define the flow of information through the network
    def forward(self, x):
        # Compute the output feature map from the convolutional layers
        out = self.get_conv_output(x)

        # Layer 5: FC + Batch Norm
        fc5_out = self.fc5(out[self.CONV_OUTPUT])
        bn5_out = self.bn5(fc5_out)

        # Linear FC layer, outputs are the class scores
        fc6_out = self.fc6(bn5_out).view(-1, NUM_CLASSES)

        # Build dictionary containing outputs from convolutional and FC layers
        out[self.FC5] = fc5_out
        out[self.BN5] = bn5_out
        out[self.FC6] = fc6_out
        return out


def unfold_map2d(input, kernel_height, kernel_width):
    # Before performing an operation between an input and a sliding kernel we need to unfold the input, i.e. the
    # windows on which the kernel is going to be applied are extracted and set apart. For this purpose, the kernel
    # shape is passed as argument to the operation. The single extracted windows are reshaped by the unfold operation
    # to rank 1 vectors. The output of F.unfold(input, (kernel_height, kernel_width)).transpose(1, 2) is a
    # tensor structured as follows: the first dimension is the batch dimension; the second dimension is the slide
    # dimension, i.e. each element is a window extracted at a different offset (and reshaped to a rank 1 vector);
    # the third dimension is a scalar within said vector.
    inp_unf = F.unfold(input, (kernel_height, kernel_width)).transpose(1, 2)
    # Now we need to reshape our tensors to the actual shape that we want in output, which is the following: the
    # first dimension is the batch dimension, the second dimension is the output channels dimension, the third and
    # fourth are height and width dimensions (obtained by splitting the former third dimension, the slide dimension,
    # representing a linear offset within the input map, into two new dimensions representing height and width), the
    # fifth is the window components dimension, corresponding to the elements of a window extracted from the input with
    # the unfold operation (reshaped to rank 1 vectors). The resulting tensor
    # is then returned.
    inp_unf = inp_unf.view(
        input.size(0),  # Batch dimension
        1,  # Output channels dimension
        input.size(2) - kernel_height + 1,  # Height dimension
        input.size(3) - kernel_width + 1,  # Width dimension
        -1  # Filter/window dimension
    )
    return inp_unf

# Custom vectorial function representing sum of an input with a sliding kernel, just like convolution is multiplication
# by a sliding kernel (as an analogy think convolution as a kernel_mult2d)


def kernel_sum2d(input, kernel):
    # In order to perform the sum with the sliding kernel we first need to unfold the input. The resulting tensor will
    # have the following structure: the first dimension is the batch dimension, the second dimension is the output
    # channels dimension, the third and fourth are height and width dimensions, the fifth is the filter/window
    # components dimension, corresponding to the elements of a window extracted from the input with the unfold
    # operation and equivalently to the elements of a filter (reshaped to rank
    # 1 vectors)
    inp_unf = unfold_map2d(input, kernel.size(2), kernel.size(3))
    # At this point the two tensors can be summed. The kernel is reshaped by unsqueezing singleton dimensions along
    # the batch dimension and the height and width dimensions. By exploiting broadcasting, it happens that the inp_unf
    # tensor is broadcast over the output channels dimension (since its shape along this dimension is 1) and therefore
    # it is automatically processed against the different filters of the kernel. In the same way, the kernel is
    # broadcast along the first dimension (and thus automatically processed against the different inputs along
    # the batch dimension) and along the third and fourth dimensions (and thus automatically processed against
    # different windows extracted from the image at different height and width
    # offsets).
    out = inp_unf + kernel.view(1, kernel.size(0), 1, 1, -1)
    return out

# Test the implementation of the kernel_sum2d function


def test_kernelsum():
    x = torch.randn(
        8,  # Batch dimension
        3,  # Input channels dimension
        10,  # Height dimension
        12  # Width dimension
    )
    w = torch.randn(
        6,  # Output channels dimension
        3,  # Input channels dimension
        4,  # Height dimension
        5   # Width dimension
    )
    output = torch.empty(
        x.shape[0],  # Batch dimension
        w.shape[0],  # Output channels dimension
        x.shape[2] - w.shape[2] + 1,  # Height dimension
        x.shape[3] - w.shape[3] + 1,  # Width dimension
        w.shape[1] * w.shape[2] * w.shape[3]  # Filter dimension
    )

    # Cross-validate vectorial implementation with for-loop implementation
    for batch in range(0, x.shape[0]):  # Loop over batch dimension
        for outchn in range(
                0, w.shape[0]):  # Loop over output channel dimension
            for i in range(0, x.shape[2] - w.shape[2] +
                           1):  # Loop over height dimension
                for j in range(
                        0, x.shape[3] - w.shape[3] + 1):  # Loop over width dimension
                    output[batch, outchn, i, j, :] = (
                        x[batch, :, i:i + w.shape[2], j:j + w.shape[3]] + w[outchn, :, :, :]).view(-1)

    out = kernel_sum2d(x, w)

    print((output.equal(out)))  # Should print out True


# Compute product between input and sliding kernel
def kernel_mult2d(x, w, b=None):
    return F.conv2d(x, w, b)

# Projection of input on weight vectors


def vector_proj2d(x, w, bias=None):
    # Compute scalar product with sliding kernel
    prod = kernel_mult2d(x, w)
    # Divide by the norm of the weight vector to obtain the projection
    norm_w = torch.norm(w.view(w.size(0), -1), p=2, dim=1).view(1, -1, 1, 1)
    norm_w += (norm_w == 0).float()  # Prevent divisions by zero
    if bias is None:
        return prod / norm_w
    return prod / norm_w + bias.view(1, -1, 1, 1)

# Projection of input on weight vector clipped between 0 and +inf


def clp_vector_proj2d(x, w, bias=None):
    return vector_proj2d(x, w, bias).clamp(0)

# Sigmoid similarity


def sig_sim2d(x, w, bias=None):
    proj = vector_proj2d(x, w, bias)
    # return torch.sigmoid((proj - proj.mean())/proj.std())
    return torch.sigmoid(proj)

# Cosine similarity between an input map and a sliding kernel


def cos_sim2d(x, w, bias=None):
    proj = vector_proj2d(x, w)
    # Divide by the norm of the input to obtain the cosine similarity
    x_unf = unfold_map2d(x, w.size(2), w.size(3))
    norm_x = torch.norm(x_unf, p=2, dim=4)
    norm_x += (norm_x == 0).float()  # Prevent divisions by zero
    if bias is None:
        return proj / norm_x
    return (proj / norm_x + bias.view(1, -1, 1, 1)).clamp(-1, 1)

# Cosine similarity clipped between 0 and 1


def clp_cos_sim2d(x, w, bias=None):
    return cos_sim2d(x, w, bias).clamp(0)

# Cosine similarity remapped to 0, 1


def raised_cos2d(x, w, bias=None):
    return (cos_sim2d(x, w, bias) + 1) / 2

# Returns function that computes raised cosine power p


def raised_cos2d_pow(p=2):
    def raised_cos2d_pow_p(x, w, bias=None):
        if bias is None:
            return raised_cos2d(x, w).pow(p)
        return (raised_cos2d(x, w).pow(p) + bias.view(1, -1, 1, 1)).clamp(0, 1)
    return raised_cos2d_pow_p

# Softmax on weight vector projection activation function


def proj_smax2d(x, w, bias=None):
    e_pow_y = torch.exp(vector_proj2d(x, w, bias))
    return e_pow_y / e_pow_y.sum(1, keepdims=True)

# Response of a gaussian activation function


def gauss(x, w, sigma=None):
    d = torch.norm(kernel_sum2d(x, -w), p=2, dim=4)
    if sigma is None:
        # heuristic: use number of dimensions as variance
        return torch.exp(-d.pow(2) / (2 * shape2size(tuple(w[0].size()))))
    # if sigma is None: return torch.exp(-d.pow(2) / (2 * torch.norm(w.view(w.size(0), 1, -1) - w.view(1, w.size(0), -1), p=2, dim=2).max().pow(2)/w.size(0))) # heuristic: normalization condition
    # if sigma is None: return torch.exp(-d.pow(2) / (2 * d.mean().pow(2)))
    return torch.exp(-d.pow(2) / (2 * (sigma.view(1, -1, 1, 1).pow(2))))

# Returns lambda function for exponentially decreasing learning rate scheduling


def sched_exp(tau=1000, eta_min=0.01):
    gamma = torch.exp(torch.tensor(-1. / tau)).item()
    return lambda eta: (eta * gamma).clamp(eta_min)


# This module represents a layer of convolutional neurons that are trained
# with a Hebbian-WTA rule
class HebbianMap2d(nn.Module):
    # Types of learning rules
    RULE_BASE = 'base'  # delta_w = eta * lfb * (x - w)
    RULE_HEBB = 'hebb'  # delta_w = eta * y * lfb * (x - w)

    # Types of LFB kernels
    LFB_GAUSS = 'gauss'
    LFB_DoG = 'DoG'
    LFB_EXP = 'exp'
    LFB_DoE = 'DoE'

    def __init__(self,
                 in_channels,
                 out_size,
                 kernel_size,
                 competitive=True,
                 random_abstention=False,
                 lfb_value=0,
                 similarity=raised_cos2d_pow(2),
                 out=vector_proj2d,
                 weight_upd_rule=RULE_BASE,
                 eta=0.1,
                 lr_schedule=None,
                 tau=1000):
        super(HebbianMap2d, self).__init__()

        # Init weights
        out_size_list = [out_size] if not hasattr(
            out_size, '__len__') else out_size
        self.out_size = torch.tensor(
            out_size_list[0:min(len(out_size_list), 3)])
        out_channels = self.out_size.prod().item()
        if hasattr(kernel_size, '__len__') and len(kernel_size) == 1:
            kernel_size = kernel_size[0]
        if not hasattr(kernel_size, '__len__'):
            kernel_size = [kernel_size, kernel_size]
        stdv = 1 / (in_channels * kernel_size[0] * kernel_size[1]) ** 0.5
        self.register_buffer(
            'weight',
            torch.empty(
                out_channels,
                in_channels,
                kernel_size[0],
                kernel_size[1]))
        # Same initialization used by default pytorch conv modules (the one
        # from the paper "Efficient Backprop, LeCun")
        nn.init.uniform_(self.weight, -stdv, stdv)

        # Enable/disable features as random abstention, competitive learning,
        # lateral feedback
        self.competitive = competitive
        self.random_abstention = competitive and random_abstention
        self.lfb_on = competitive and isinstance(lfb_value, str)
        self.lfb_value = lfb_value

        # Set output function, similarity function and learning rule
        self.similarity = similarity
        self.out = out
        self.teacher_signal = None  # Teacher signal for supervised training
        self.weight_upd_rule = weight_upd_rule

        # Initial learning rate and lR scheduling policy. LR wrapped into a
        # registered buffer so that we can save/load it
        self.register_buffer('eta', torch.tensor(eta))
        self.lr_schedule = lr_schedule  # LR scheduling policy

        # Set parameters related to the lateral feedback feature
        if self.lfb_on:
            # Prepare the variables to generate the kernel that will be used to
            # apply lateral feedback
            map_radius = (self.out_size - 1) // 2
            sigma_lfb = map_radius.max().item()
            x = torch.abs(
                torch.arange(
                    0,
                    self.out_size[0].item()) -
                map_radius[0])
            for i in range(1, self.out_size.size(0)):
                x_new = torch.abs(
                    torch.arange(
                        0,
                        self.out_size[i].item()) -
                    map_radius[i])
                for j in range(i):
                    x_new = x_new.unsqueeze(j)
                # max gives L_infinity distance, sum would give L_1 distance,
                # root_p(sum x^p) for L_p
                x = torch.max(x.unsqueeze(-1), x_new)
            # Store the kernel that will be used to apply lateral feedback in a
            # registered buffer
            if lfb_value == self.LFB_EXP or lfb_value == self.LFB_DoE:
                self.register_buffer(
                    'lfb_kernel', torch.exp(-x.float() / sigma_lfb))
            else:
                self.register_buffer(
                    'lfb_kernel', torch.exp(-x.pow(2).float() / (2 * (sigma_lfb ** 2))))
            # Padding that will pad the inputs before applying the lfb kernel
            pad_pre = map_radius.unsqueeze(1)
            pad_post = (self.out_size - 1 - map_radius).unsqueeze(1)
            self.pad = tuple(
                torch.cat((pad_pre, pad_post), dim=1).flip(0).view(-1))
            # LFB kernel shrinking parameter
            self.alpha = torch.exp(
                torch.log(
                    torch.tensor(sigma_lfb).float()) /
                tau).item()
            if lfb_value == self.LFB_GAUSS or lfb_value == self.LFB_DoG:
                self.alpha = self.alpha ** 2
        else:
            self.register_buffer('lfb_kernel', None)

        # Init variables for statistics collection
        if self.random_abstention:
            self.register_buffer('victories_count', torch.zeros(out_channels))
        else:
            self.register_buffer('victories_count', None)

    def set_teacher_signal(self, y):
        self.teacher_signal = y

    def forward(self, x):
        y = self.out(x, self.weight)
        # print(self.training)
        if self.training:
            self.update(x)
        return y

    def update(self, x):
        # Prepare the inputs
        y = self.similarity(x, self.weight)
        t = self.teacher_signal
        if t is not None:
            t = t.unsqueeze(2).unsqueeze(
                3) * torch.ones_like(y, device=y.device)
        y = y.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
        if t is not None:
            t = t.permute(0, 2, 3, 1).contiguous(
            ).view(-1, self.weight.size(0))
        x_unf = unfold_map2d(x, self.weight.size(2), self.weight.size(3))
        x_unf = x_unf.permute(
            0, 2, 3, 1, 4).contiguous().view(
            y.size(0), 1, -1)

        # Random abstention
        if self.random_abstention:
            abst_prob = self.victories_count / \
                (self.victories_count.max() + y.size(0) / y.size(1)).clamp(1)
            scores = y * (torch.rand_like(abst_prob,
                                          device=y.device) >= abst_prob).float().unsqueeze(0)
        else:
            scores = y

        # Competition. The returned winner_mask is a bitmap telling where a
        # neuron won and where one lost.
        if self.competitive:
            if t is not None:
                scores *= t
            winner_mask = (scores == scores.max(1, keepdim=True)[0]).float()
            if self.random_abstention:  # Update statistics if using random abstension
                # Number of inputs over which a neuron won
                winner_mask_sum = winner_mask.sum(0)
                self.victories_count += winner_mask_sum
                self.victories_count -= self.victories_count.min().item()
        else:
            winner_mask = torch.ones_like(y, device=y.device)

        # Lateral feedback
        if self.lfb_on:
            lfb_kernel = self.lfb_kernel
            if self.lfb_value == self.LFB_DoG or self.lfb_value == self.LFB_DoE:
                # Difference of Gaussians/Exponentials (mexican hat shaped
                # function)
                lfb_kernel = 2 * lfb_kernel - lfb_kernel.pow(0.5)
            lfb_in = F.pad(winner_mask.view(-1, *self.out_size), self.pad)
            if self.out_size.size(0) == 1:
                lfb_out = torch.conv1d(
                    lfb_in.unsqueeze(1),
                    lfb_kernel.unsqueeze(0).unsqueeze(1))
            elif self.out_size.size(0) == 2:
                lfb_out = torch.conv2d(
                    lfb_in.unsqueeze(1),
                    lfb_kernel.unsqueeze(0).unsqueeze(1))
            else:
                lfb_out = torch.conv3d(
                    lfb_in.unsqueeze(1),
                    lfb_kernel.unsqueeze(0).unsqueeze(1))
            lfb_out = lfb_out.clamp(-1, 1).view_as(y)
        else:
            lfb_out = winner_mask
            if self.competitive:
                lfb_out[lfb_out == 0] = self.lfb_value
            elif t is not None:
                lfb_out = t

        # Compute step modulation coefficient
        r = lfb_out  # RULE_BASE
        if self.weight_upd_rule == self.RULE_HEBB:
            r *= y

        # Compute delta
        r_abs = r.abs()
        r_sign = r.sign()
        delta_w = r_abs.unsqueeze(
            2) * (r_sign.unsqueeze(2) * x_unf - self.weight.view(1, self.weight.size(0), -1))

        # Since we use batches of inputs, we need to aggregate the different update steps of each kernel in a unique
        # update. We do this by taking the weighted average of teh steps, the weights being the r coefficients that
        # determine the length of each step
        r_sum = r_abs.sum(0)
        r_sum += (r_sum == 0).float()  # Prevent divisions by zero
        delta_w_avg = (delta_w * r_abs.unsqueeze(2)
                       ).sum(0) / r_sum.unsqueeze(1)

        # Apply delta
        self.weight += self.eta * delta_w_avg.view_as(self.weight)

        # LFB kernel shrinking and LR schedule
        if self.lfb_on:
            self.lfb_kernel = self.lfb_kernel.pow(self.alpha)
        if self.lr_schedule is not None:
            self.eta = self.lr_schedule(self.eta)


# Generate a batch of random inputs for testing
def gen_batch(centers, batch_size, win_height, win_width):
    # Generate an input "image" by first generating patches as random perturbations on the cluster centers and then
    # concatenating them in the horizontal and vertical dimensions. Repeat to
    # generate a batch.
    batch = torch.empty(0)
    for j in range(batch_size):  # Loop to generate batch
        image = torch.empty(0)
        for k in range(win_height):  # Loop to concat image rows vertically
            row = torch.empty(0)
            for l in range(win_width):  # Loop to concat patches horizontally
                # Generate an input patch by perturbing a cluster center
                index = int(
                    torch.floor(
                        torch.rand(1) *
                        centers.size(0)).item())
                patch = centers[index] + 0.1 * torch.randn_like(centers[index])
                # Concatenate patch horizonally to the image row
                row = torch.cat((row, patch), 2)
            # Concatenate row to the image vertically
            image = torch.cat((image, row), 1)
        # Concatenate the image to the batch
        batch = torch.cat((batch, image.unsqueeze(0)), 0)
    return batch

# Test for the batch generation function


def test_genbatch():
    # Generate centers around which clusters are built
    centers = torch.randn(6, 3, 4, 5)
    # Generate a batch of inputs around the centers
    batch = gen_batch(centers, 10, 2, 2)
    # Check that the batch size is correct (just to be sure)
    print(batch.size())  # Should print 10x3x8x10


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.


def get_model():
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :return: the model instance
    """
    # init the model and the preprocessing:
    preprocessing = functools.partial(
        load_preprocess_images, image_size=img_size)
    
    mynet = Net()
    
    # get an activations model from the Pytorch Wrapper
    activations_model = PytorchWrapper(
        identifier=MODEL_NAME,
        model=mynet,
        preprocessing=preprocessing)

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = img_size
    return wrapper


# Main Method: In submitting a custom model, you should not have to mess
# with this.
if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
