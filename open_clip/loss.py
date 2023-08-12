import torch
import torch.nn as nn
from torch.nn import functional as F

import hashlib
from oc_utils.custom_loss.sup_con_loss import SupConLoss

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def myStringHash(s, byte_length=7):
    # https://ryansblog.xyz/post/3a1b00b7-e4a5-4f0a-9742-05f3171ad317
    # NOTE: sha256 returns a 256 bit int, but torch.tensor can only handle 64 bit signed int
    # therefore, we truncate to 7 bytes < 63 bits
    v = hashlib.sha256(s.encode("utf-8"))
    return int.from_bytes(v.digest()[:byte_length], 'little')


def gather_labels(
    labels,
    world_size=1,
):
    output = [torch.zeros_like(labels) for _ in range(world_size)]
    dist.all_gather(output, labels)
    return torch.cat(tuple(output))


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipSupConLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            bigbatch=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.bigbatch = bigbatch
        self.sup_con_loss = SupConLoss()

    def forward(self, image_features, text_features, logit_scale, target_values):
        device = image_features.device
        if self.bigbatch and self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_text = logit_scale * text_features @ image_features.T
            logits_per_image = logit_scale * image_features @ text_features.T

        # Calculated ground-truth
        if self.bigbatch and self.world_size > 1:
            # Convert string labels into 64 bit integers by hashing
            target_values = torch.tensor(
                [abs(myStringHash(v)) for v in target_values], dtype=torch.int64, device=image_features.device)
            all_labels = gather_labels(target_values, self.world_size).contiguous().view(-1, 1)
        else:
            label_dict = {val: i for i, val in enumerate(list(dict.fromkeys(target_values)))}
            all_labels = torch.tensor([label_dict[val] for val in target_values], device=device).contiguous().view(-1, 1)

        # Get Sup Con loss
        image_loss = self.sup_con_loss(logits_per_image, labels=all_labels)
        text_loss = self.sup_con_loss(logits_per_text, labels=all_labels)
        total_loss = (image_loss + text_loss) / 2

        return total_loss


class ClipInfonceLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            bigbatch=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.bigbatch = bigbatch

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, target_values): # "target_values" is not used for InfonceLoss
        device = image_features.device
        if self.bigbatch and self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.bigbatch and self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss
