"""The MTP loss, alone or on top of sampled passes (lorm.train.algo.SampledPasses). The MTP heads' losses share
instance.mtp_weight, and the "state_alignment" stand-in adds instance.alignment_weight times its own.
"""

from lorm.registry import ALGORITHMS
from lorm.train.algo import Algorithm, SampledPasses, training_flops_per_token
from lorm.train.loss import causal_loss


def mtp_loss(config, output, labels):
    """Head k's logits at t predict token t + k + 1."""
    instance = config.instance
    loss = instance.mtp_weight / len(output.logits) * sum(causal_loss(logits, labels[:, k:])
                                                         for k, logits in enumerate(output.logits, 1))
    return loss if output.stand_in is None else loss + instance.alignment_weight * output.stand_in


class MTPAlgorithm(Algorithm):
    """The next-token loss plus the MTP loss, for the Transformer with MTP heads."""

    def loss(self, model, inputs, labels):
        logits, output = model(inputs, mtp=True)
        return causal_loss(logits, labels) + mtp_loss(getattr(model, "module", model).config, output, labels)

    def flops_per_token(self, model, sequence_length):
        return training_flops_per_token(model, sequence_length, mtp=True)


class MTPSampledPasses(SampledPasses):
    """Sampled passes (lorm.train.algo) plus the MTP loss and the stand-in's loss."""

    def loss(self, model, inputs, labels):
        config = getattr(model, "module", model).config
        logits, output = model(inputs, passes=self.passes, mtp=True)
        return causal_loss(logits, labels) + mtp_loss(config, output, labels)

    def flops_per_token(self, model, sequence_length):
        return training_flops_per_token(model, sequence_length, passes=self.passes, mtp=True)


ALGORITHMS["mtp"] = MTPAlgorithm
ALGORITHMS["mtp_sampled_passes"] = MTPSampledPasses
