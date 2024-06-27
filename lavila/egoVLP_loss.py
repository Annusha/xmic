import torch
import torch.nn.functional as F
from torch import nn




class EgoNCE(nn.Module):
    def __init__(self, temperature=0.07, noun=True, verb=True):
        super().__init__()
        self.noun = noun
        self.verb = verb
        self.temperature = temperature

    def forward(self, t2v, v2t, mask_v, mask_n, multi_pad_mask, strict_mask=False, vn_threshold=0):
        # multiple positive sample
        temperature = 0.7

        masked_t2v = t2v.masked_fill(~multi_pad_mask.bool(), float('-inf'))

        # create diagonal mask for postive text-video pairs
        multi_pos_mask = torch.eye(t2v.shape[-1], device=t2v.device)[:, None, :]
        R = multi_pad_mask.shape[0] // multi_pad_mask.shape[1]  # how many captions per video
        multi_pos_mask = multi_pos_mask.repeat(1, R, 1).flatten(0, 1)

        # add more positives to the mask -- sentences that share the same verb and nouns as positives
        if mask_v is not None and mask_n is not None:
            mask_v2 = mask_v[:, None, :].repeat(1, R, 1).flatten(0, 1)
            mask_n2 = mask_n[:, None, :].repeat(1, R, 1).flatten(0, 1)

            mask_vn = (mask_v2 * mask_n2)
            mask_pos = (multi_pos_mask)
            mask = (mask_v2 * mask_n2 + mask_pos)

            mask_vn = mask_vn[masked_t2v.sum(-1) != float('-inf')]
            mask_pos = mask_pos[masked_t2v.sum(-1) != float('-inf')]
            mask = mask[masked_t2v.sum(-1) != float('-inf')]

        elif mask_n is not None:
            mask_n = mask_n[:, None, :].repeat(1, 5, 1).flatten(0, 1)
            mask = (mask_n + multi_pos_mask) * multi_pad_mask

            mask = mask[masked_t2v.sum(-1) != float('-inf')]
        elif mask_v is not None:
            mask_v = mask_v[:, None, :].repeat(1, 5, 1).flatten(0, 1)
            mask = (mask_v + multi_pos_mask) * multi_pad_mask
            mask = mask[masked_t2v.sum(-1) != float('-inf')]

        # breakpoint()
        masked_v2t = v2t[:, masked_t2v.sum(-1) != float('-inf')]
        masked_t2v2 = masked_t2v[masked_t2v.sum(-1) != float('-inf')]

        mask_bool = mask > vn_threshold

        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_sm = masked_t2v2 / temperature
        idiag = torch.sum(torch.log_softmax(i_sm, dim=1) * mask_bool, dim=1)
        idiag = idiag / mask_bool.sum(-1)
        loss_i = idiag.sum() / len(idiag)

        j_sm = masked_v2t / temperature
        jdiag = torch.log_softmax(j_sm, dim=1) * mask_bool.t()
        jdiag = jdiag.sum(1)
        # jdiag = torch.sum(torch.log_softmax(j_sm, dim=1) * mask_bool.t(), dim=1)
        jdiag = jdiag / mask_bool.sum(0)
        loss_j = jdiag.sum() / len(jdiag)
        return - loss_i - loss_j, {'loss_v2t':loss_j, 'loss_t2v': loss_i}
        # return - loss_i, mask_bool



def EgoNCE_loss( t2v, v2t, mask_v, mask_n, multi_pad_mask, strict_mask=False, vn_threshold=0):
    # multiple positive sample
    temperature = 0.7

    masked_t2v = t2v.masked_fill(~multi_pad_mask.bool(), float('-inf'))

    # create diagonal mask for postive text-video pairs
    multi_pos_mask = torch.eye(t2v.shape[-1], device=t2v.device)[:, None, :]
    R = multi_pad_mask.shape[0] // multi_pad_mask.shape[1]  # how many captions per video
    multi_pos_mask = multi_pos_mask.repeat(1, R, 1).flatten(0, 1)

    # add more positives to the mask -- sentences that share the same verb and nouns as positives
    if mask_v is not None and mask_n is not None:
        mask_v2 = mask_v[:, None, :].repeat(1, R, 1).flatten(0, 1)
        mask_n2 = mask_n[:, None, :].repeat(1, R, 1).flatten(0, 1)

        mask_vn = (mask_v2 * mask_n2)
        mask_pos = (multi_pos_mask)
        mask = (mask_v2 * mask_n2 + mask_pos)

        mask_vn = mask_vn[masked_t2v.sum(-1) != float('-inf')]
        mask_pos = mask_pos[masked_t2v.sum(-1) != float('-inf')]
        mask = mask[masked_t2v.sum(-1) != float('-inf')]

    elif mask_n is not None:
        mask_n = mask_n[:, None, :].repeat(1, 5, 1).flatten(0, 1)
        mask = (mask_n + multi_pos_mask) * multi_pad_mask

        mask = mask[masked_t2v.sum(-1) != float('-inf')]
    elif mask_v is not None:
        mask_v = mask_v[:, None, :].repeat(1, 5, 1).flatten(0, 1)
        mask = (mask_v + multi_pos_mask) * multi_pad_mask
        mask = mask[masked_t2v.sum(-1) != float('-inf')]

    # breakpoint()
    masked_v2t = v2t[:, masked_t2v.sum(-1) != float('-inf')]
    masked_t2v2 = masked_t2v[masked_t2v.sum(-1) != float('-inf')]

    mask_bool = mask > vn_threshold

    "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
    i_sm = masked_t2v2 / temperature
    idiag = torch.sum(torch.log_softmax(i_sm, dim=1) * mask_bool, dim=1)
    idiag = idiag / mask_bool.sum(-1)
    loss_i = idiag.sum() / len(idiag)

    j_sm = masked_v2t / temperature
    jdiag = torch.log_softmax(j_sm, dim=1) * mask_bool.t()
    jdiag = jdiag.sum(1)
    # jdiag = torch.sum(torch.log_softmax(j_sm, dim=1) * mask_bool.t(), dim=1)
    jdiag = jdiag / mask_bool.sum(0)
    loss_j = jdiag.sum() / len(jdiag)
    return - loss_i - loss_j, {'loss_v2t':loss_j, 'loss_t2v': loss_i}