"""
calcute acc

"""

from sklearn.metrics import accuracy_score


def top1_acc(pred, targets, **kwargs):
    """
    cal top1 acc
    Args:
        pred: array or paddle.tensor, (n, c)
        targets: array or paddle.tensor, (n,)

    """
    assert len(pred.shape) == 2, "dim of predict should be 2, (n x c)"
    ema_pred = kwargs['ema_pred']
    pred_y = pred.argmax(axis=-1)
    
    p = pred.argmax(axis=-1)
    ema_acc = None
    if ema_pred is not None:
        ema_p = ema_pred.argmax(axis=-1)
        ema_acc = accuracy_score(ema_p, targets)
    results = {'acc': accuracy_score(p, targets), 'ema_acc': ema_acc}
    return results


def top1_pseudo_acc(pseudo_pred, pseudo_targets, threshold=0, **kargs):
    """
    cal top1 acc
    Args:
        pred: array or paddle.tensor, (n, c)
        targets: array or paddle.tensor, (n,)

    """
    assert len(pseudo_pred.shape) == 2, "dim of predict should be 2, (n x c)"
    max_probs = pseudo_pred.max(axis=-1)
    mask = (max_probs > threshold).astype('float32')
    pred_y = pseudo_pred.argmax(axis=-1)
    right_pred = (pred_y == pseudo_targets).astype('float32') * mask
    acc = right_pred.sum() / max(mask.sum(), 1.)

    # acc = pred.argmax(axis=-1)
    # return accuracy_score(acc, targets)
    results = {'top1_pseudo_acc': round(acc.item(), 4),
               'mask_prob': round(mask.mean().item(), 4)}
    return results

