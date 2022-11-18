"""seq2seq for training."""
import numpy as np

from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore import Parameter
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.communication.management import get_group_size

from .seq2seq import Seq2seqModel


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 5.0

clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad

class PredLogProbs(nn.Cell):
    """
    Get log probs.

    Args:
        config (GNMTConfig): The config of GNMT.

    Returns:
        Tensor, log softmax output.
    """

    def __init__(self):
        super(PredLogProbs, self).__init__()
        self.reshape = P.Reshape()
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.get_shape = P.Shape()

    def construct(self, input_tensor):
        """
        Construct network.

        Args:
            input_tensor (Tensor): Tensor.

        Returns:
            Tensor, log softmax output.
        """
        shape = self.get_shape(input_tensor)
        logits = self.reshape(input_tensor, (shape[0] * shape[1], shape[2]))
        log_probs = self.log_softmax(logits)
        return log_probs


class LabelSmoothedCrossEntropyCriterion(nn.Cell):
    """
    Label Smoothed Cross-Entropy Criterion.

    Args:
        config (Seq2seqConfig): The config of Seq2seq.

    Returns:
        Tensor, final loss.
    """

    def __init__(self, config):
        super(LabelSmoothedCrossEntropyCriterion, self).__init__()
        self.vocab_size = config.vocab_size
        self.batch_size = config.batch_size
        self.smoothing = 0.1
        self.confidence = 0.9
        self.last_idx = (-1,)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.index_ids = Tensor(np.arange(config.batch_size * config.max_decode_length).reshape((-1, 1)), mstype.int32)
        self.gather_nd = P.GatherNd()
        self.expand = P.ExpandDims()
        self.concat = P.Concat(axis=-1)

    def construct(self, prediction_scores, label_ids, label_weights):
        """
        Construct network to calculate loss.

        Args:
            prediction_scores (Tensor): Prediction scores. [batchsize, seq_len, vocab_size]
            label_ids (Tensor): Labels. [batchsize, seq_len]
            label_weights (Tensor): Mask tensor. [batchsize, seq_len]

        Returns:
            Tensor, final loss.
        """
        prediction_scores = self.reshape(prediction_scores, (-1, self.vocab_size))
        label_ids = self.reshape(label_ids, (-1, 1))
        label_weights = self.reshape(label_weights, (-1,))
        tmp_gather_indices = self.concat((self.index_ids, label_ids))
        nll_loss = self.neg(self.gather_nd(prediction_scores, tmp_gather_indices))
        nll_loss = label_weights * nll_loss
        smooth_loss = self.neg(self.reduce_mean(prediction_scores, self.last_idx))
        smooth_loss = label_weights * smooth_loss
        loss = self.reduce_sum(self.confidence * nll_loss + self.smoothing * smooth_loss, ())
        loss = loss / self.batch_size
        return loss


class Seq2seqTraining(nn.Cell):
    """
    seq2seq training network.

    Args:
        config (seq2seqConfig): The config of seq2seq.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings.

    Returns:
        Tensor, prediction_scores.
    """

    def __init__(self, config, is_training, use_one_hot_embeddings):
        super(Seq2seqTraining, self).__init__()
        self.seq2seq = Seq2seqModel(config, is_training, use_one_hot_embeddings)
        self.projection = PredLogProbs()

    def construct(self, source_ids, source_mask, target_ids):
        """
        Construct network.

        Args:
            source_ids (Tensor): Source sentence.
            source_mask (Tensor): Source padding mask.
            target_ids (Tensor): Target sentence.

        Returns:
            Tensor, prediction_scores.
        """
        decoder_outputs = self.seq2seq(source_ids, source_mask, target_ids)
        prediction_scores = self.projection(decoder_outputs)
        return prediction_scores


class Seq2seqNetworkWithLoss(nn.Cell):
    """
    Provide  seq2seq training loss through network.

    Args:
        config (seq2seqconfig): The config of seq2seq.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(Seq2seqNetworkWithLoss, self).__init__()
        self.seq2seq = Seq2seqTraining(config, is_training, use_one_hot_embeddings)
        self.loss = LabelSmoothedCrossEntropyCriterion(config)
        self.cast = P.Cast()

    def construct(self,
                  source_ids,
                  source_mask,
                  target_ids,
                  label_ids,
                  label_weights):
        prediction_scores = self.seq2seq(source_ids, source_mask, target_ids)
        total_loss = self.loss(prediction_scores, label_ids, label_weights)
        return self.cast(total_loss, mstype.float32)


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)

class Seq2seqTrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of seq2seq network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network: Cell. The training network. Note that loss function should have
            been added.
        optimizer: Optimizer. Optimizer for updating the weights.

    Returns:
        Tuple[Tensor, Tensor, Tensor], loss, overflow, sen.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(Seq2seqTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)

        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self,
                  source_eos_ids,
                  source_eos_mask,
                  target_sos_ids,
                  target_eos_ids,
                  target_eos_mask,
                  sens=None):
        """
        Construct network.

        Args:
            source_eos_ids (Tensor): Source sentence.
            source_eos_mask (Tensor): Source padding mask.
            target_sos_ids (Tensor): Target sentence.
            target_eos_ids (Tensor): Prediction sentence.
            target_eos_mask (Tensor): Prediction padding mask.
            sens (Tensor): Loss sen.

        Returns:
            Tuple[Tensor, Tensor, Tensor], loss, overflow, sen.
        """
        source_ids = source_eos_ids
        source_mask = source_eos_mask
        target_ids = target_sos_ids
        label_ids = target_eos_ids
        label_weights = target_eos_mask

        weights = self.weights
        loss = self.network(source_ids,
                            source_mask,
                            target_ids,
                            label_ids,
                            label_weights)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        grads = self.grad(self.network, weights)(source_ids,
                                                 source_mask,
                                                 target_ids,
                                                 label_ids,
                                                 label_weights,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        ret = (loss, cond, scaling_sens)
        return ret
