import torch.nn as nn
import torch.nn.functional as F

def distill_loss(student_logits,teacher_logits,T=2.0):
    kldiv = nn.KLDivLoss(reduction="batchmean",log_target=True)
    student_log_softmax = F.log_softmax(student_logits/T,dim=1)
    teacher_log_softmax = F.log_softmax(teacher_logits/T,dim=1)
    loss = kldiv(student_log_softmax,teacher_log_softmax) * (T**2)
    return loss