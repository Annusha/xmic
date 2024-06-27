from collections import defaultdict
import torch
import numpy as np

__all__ = ["AverageMeter", "MetricMeter", "lt_accuracies", "lt_accuracies_splitted"]


class AverageMeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


class MetricMeter:
    """Store the average and current value for a set of metrics.

    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, delimiter=" "):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                "Input to MetricMeter.update() must be a dictionary"
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(f"{name} {meter.val:.4f} ({meter.avg:.4f})")
        return self.delimiter.join(output_str)


def lt_accuracies(preds, labels, n_classes, dist_splits, label_type='orig'):
    """
    Calculates avg class accuracy for all classes, head, tail and few-shot tail
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
    """
    # n_classes = len(train_class_counts)
    if label_type != 'orig':
        n_classes = n_classes[label_type]
        new_dist_split = []
        for subsplits in dist_splits:
            subsplit = subsplits[label_type]
            new_dist_split.append(subsplit)
        dist_splits = new_dist_split

    if len(preds.shape) == 2:
        correct = torch.eq(torch.argmax(preds, dim=-1), labels).float()
    else:
        correct = torch.eq(preds, labels).float()

    test_counts = {}
    test_correct = {}
    train_counts = {}
    test_class_acc = {}

    test_count_list = torch.zeros(n_classes, device=preds.device)
    test_correct_list = torch.zeros(n_classes, device=preds.device)

    for l in torch.unique(labels):
        if torch.is_tensor(l):
            l = l.item()
        # train_counts[l] = train_class_counts[l]

        l = int(l)
        count = torch.sum(torch.where(labels == l, 1, 0))
        test_counts[l] = count
        test_count_list[l] = count

        l_correct = (correct * (labels == l)).float().sum()
        test_correct[l] = l_correct
        test_correct_list[l] = l_correct
        test_class_acc[l] = test_correct[l] / test_counts[l]

    avg_accs = []
    avg_accs.append(torch.mean(torch.stack([a for a in test_class_acc.values()])) * 100)
    full_accs = []
    full_accs.append(torch.sum(test_correct_list) / torch.sum(test_count_list) * 100)

    for split in dist_splits:
        split_accs = []
        split_correct_count = 0
        split_count = 0
        for k in test_class_acc.keys():
            if k in split:
                split_accs.append(test_class_acc[k])
                split_correct_count += test_correct[k]
                split_count += test_counts[k]
        if len(split_accs) > 0:
            split_accs = torch.mean(torch.stack(split_accs))
            split_accs_full = split_correct_count / split_count
        else:
            split_accs = torch.tensor(-1, device=preds.device)
            split_accs_full = -1

        avg_accs.append(split_accs * 100)
        full_accs.append(split_accs_full * 100)

    avg_accs = torch.tensor(avg_accs, device=preds.device)
    full_accs = torch.tensor(full_accs, device=preds.device)

    test_class_acc = {k: float(v.detach().cpu().numpy()) for k, v, in test_class_acc.items()}
    # print(train_counts)
    # train_counts = {k: v for k, v, in train_counts.items()}

    return {
        'avg_accs': avg_accs,
        'test_correct_list': test_correct_list,
        'test_count_list': test_count_list,
        'test_class_acc': test_class_acc,
        'full_accs': full_accs,
        # 'train_counts': train_counts
    }

def lt_accuracies_splitted(preds, labels, dist_splits, label_type='orig'):
    """
    Calculates avg class accuracy for all classes, head, tail and few-shot tail
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
    """
    if label_type != 'orig':
        new_dist_split = []
        for subsplits in dist_splits:
            subsplit = subsplits[label_type]
            new_dist_split.append(subsplit)
        dist_splits = new_dist_split

    assert len(preds.shape) == 2

    full_accs = []

    for split in dist_splits:
        mask_split = torch.zeros(preds.shape[0], dtype=bool)
        mask_classes = torch.ones(preds.shape[1], dtype=bool)
        for k in split:
            mask_split += labels == k
            mask_classes[k] = False

        # breakpoint()
        preds_split = preds[mask_split]
        labels_split = labels[mask_split]
        preds_split[:, mask_classes] = -np.inf

        correct = torch.eq(torch.argmax(preds_split, dim=-1), labels_split).float()
        if len(correct) == 0:
            full_accs.append(0)
        else:
            full_accs.append(correct.sum() / len(correct) * 100)

    full_accs = torch.tensor(full_accs, device=preds.device)


    return {
        'full_accs': full_accs,
    }
