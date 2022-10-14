from torch.nn import CrossEntropyLoss
import torch

# the loss function of symptom inquiry prediction with learning wieght
def lm_loss_func(lm_logits, sym_labels, decoder_pos: torch.Tensor, decoder_weight, sym_mask, class_weight=None):
    # loss_fct = CrossEntropyLoss(weight=class_weight ,ignore_index=-1,reduction='none')
    loss_fct = CrossEntropyLoss(ignore_index=-1,reduction='none')
    decoder_pos_index = decoder_pos.unsqueeze(-1)
    decoder_pos_index = decoder_pos_index.expand((-1,)*(decoder_pos_index.dim()-1)+(lm_logits.size(-1),))
    lm_logits = lm_logits.gather(-2,decoder_pos_index)

    lm_logits = lm_logits[decoder_pos.ne(0)]
    
    # Concurrent Softmax mask
    sym_mask = sym_mask * -10000
    lm_logits = lm_logits + sym_mask

    # or replace to implement concurrent softmax 
    # lm_logits = torch.where(sym_mask.eq(0),lm_logits,torch.tensor(-10000.0).to(lm_logits.device))
    

    # test all equal weight
    # decoder_weight = torch.where(decoder_weight > 0, 1,0)

    lm_loss = loss_fct(lm_logits, sym_labels)
    lm_loss = lm_loss * decoder_weight
    lm_loss = lm_loss.sum()/decoder_pos_index.size(0)/3

    _, preds = lm_logits.max(dim=-1)  
    # the loss of tokens with non pad_id is averaged and the accuracy of prediction is calculated
    not_ignore = sym_labels.ne(-1)  
    num_targets = not_ignore.long().sum().item()  # count the number of non-pad_id in the target

    correct = (sym_labels == preds) & not_ignore  # calculate the number of tokens that model predicts correctly
    correct = correct.float().sum()

    accuracy = correct / num_targets

    return lm_loss,accuracy


def GPT2_lm_loss_func(lm_logits, sym_labels, class_weight=None):
    loss_fct = CrossEntropyLoss(weight=class_weight,ignore_index=-1,reduction='none')
    shift_logits = lm_logits[:,1:,:].contiguous()
    loss_fct = CrossEntropyLoss(ignore_index=-1)
    lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    sym_labels.view(-1))
    
    _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]
    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    not_ignore = sym_labels.ne(-1)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

    correct = (sym_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
    correct = correct.float().sum()

    accuracy = correct / num_targets

    return lm_loss,accuracy


# the loss function of the heads of disease and encoder
def mc_loss_func(mc_logits, mc_labels, differentials=None, differentials_probs=None):
    assert (differentials is None) == (differentials_probs is None)
    # mc_logits
    loss_fct_mc = CrossEntropyLoss()
    if differentials is None:
        mc_loss = loss_fct_mc(mc_logits.view(-1, mc_logits.size(-1)),
                        mc_labels.view(-1))
    else:
        mc_loss = _soft_cross_entropy(
            mc_logits.view(-1, mc_logits.size(-1)),
            target_indices=differentials, target_probas=differentials_probs, ignore_index=-1,
        )

    # mc_accuracy
    _, mc_preds = mc_logits.max(dim=-1)
    mc_correct = (mc_preds == mc_labels).float().sum()
    mc_accuracy = mc_correct / mc_labels.size(0)

    return mc_loss, mc_accuracy

# the evaluation of symptom inquiry prediction 
def lm_test_func(lm_logits, sym_labels, decoder_pos: torch.Tensor, decoder_weight, pred_num, sep_pos, sep_token_id):

    decoder_pos = decoder_pos.unsqueeze(-1)
    decoder_pos = decoder_pos.expand((-1,)*(decoder_pos.dim()-1)+(lm_logits.size(-1),))
    sym_logoits = lm_logits.gather(-2,decoder_pos)
    _, preds = sym_logoits.max(dim=-1)  
    not_ignore = sym_labels.ne(-1)  

    correct = (sym_labels == preds) & not_ignore  
    correct = correct.float().sum()

    accuracy = correct / pred_num

    sep_pos = sep_pos.unsqueeze(-1).unsqueeze(-1)
    sep_pos = sep_pos.expand((-1,)*(sep_pos.dim()-1)+(lm_logits.size(-1),))
    sep_logoits = lm_logits.gather(-2,sep_pos)
    _, sep_preds = sep_logoits.max(dim=-1)
    sep_accuracy = sep_preds.eq(sep_token_id).float().sum() / sep_pos.size(0)

    return accuracy,sep_accuracy

def _soft_cross_entropy(
    pred, target_indices, target_probas, weight=None, reduction="mean", ignore_index=-1
):
    """Computes the cross entropy loss using soft labels.

    Here the soft labels are defined through the parameters `target_indices`
    and `target_probas`. They respectively represent the class indices involved
    in the target distribution and their corresponding probability.
    The provided `ignore_index` can be used as padding element in the `target_indices`
    field.

    Per definition, we have (https://en.wikipedia.org/wiki/Cross_entropy):
        CE(p,q) = -(p * log(q)).sum()
    With a provided weight per class, the computation becomes:
        CE(p,q,w) = -(p * log(q)).sum() * (p * w).sum()

    Parameters
    ----------
    pred: tensor
        a tensor of size `N x C x *` where N is the batch size, C is the number
        of classes, and `*` represents any other dimensions. This tensor represents
        the logit values.
    target_indices: tensor
        a tensor of size `N x D x *` where N is the batch size, D <= C is the number
        of classes present in the soft distribution, and `*` represents
        any other dimensions. It must match the tailing dimensions of `pred`.
    target_probas: tensor
        a tensor of same size as `target_indices` representing the probability
        associated to each class therein.
    weight: tensor
        a manual rescaling weight given to each class. It is a 1-D tensor of size
        `C`. Default: None
    reduction: str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        Default: 'mean'
    ignore_index: int
        Specifies a target value that is ignored and does not contribute
        to the gradient. Default: -1

    Return
    ----------
    result: tensor
        the computed loss.

    """
    assert reduction in ["none", "mean", "sum"]

    dim = pred.dim()
    if dim < 2:
        raise ValueError("Expected 2 or more dimensions (got {})".format(dim))

    dim = target_indices.dim()
    if dim < 2:
        raise ValueError("Expected 2 or more dimensions (got {})".format(dim))

    assert (weight is None) or (weight.dim() == 1 and weight.size(0) == pred.size(1))

    if pred.size(0) != target_indices.size(0):
        raise ValueError(
            f"Expected input batch_size ({pred.size(0)}) to match "
            f"target batch_size ({target_indices.size(0)})."
        )

    if pred.size(1) < target_indices.size(1):
        raise ValueError(
            f"Expected input class_size ({pred.size(1)}) to be greater/equal"
            f"than target class_size ({target_indices.size(1)})."
        )
    if target_indices.size()[2:] != pred.size()[2:]:
        out_size = target_indices.size()[:2] + pred.size()[2:]
        raise ValueError(
            f"Expected target_indices size {out_size} (got {target_indices.size()})"
        )
    if target_indices.size() != target_probas.size():
        raise ValueError(
            f"Expected target_probas size {target_indices.size()} "
            f"(got {target_probas.size()})"
        )

    log_probs = torch.nn.functional.log_softmax(pred, dim=1)
    mask = target_indices != ignore_index
    masked_indices = target_indices * mask
    tmp_weight = 1.0 if weight is None else weight[masked_indices]
    avg_log_probs = (mask * log_probs.gather(1, masked_indices) * target_probas).sum(
        dim=1
    )
    avg_weight = (
        1.0 if weight is None else (tmp_weight * mask * target_probas).sum(dim=1)
    )
    result = -(avg_weight * avg_log_probs)

    if reduction == "sum":
        result = result.sum()
    elif reduction == "mean":
        result = result.mean() if weight is None else result.sum() / avg_weight.sum()

    return result