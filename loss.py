import torch 
import torch.nn.functional as F 

def dual_kl_loss(
    student_logits, 
    honest_teacher_logits, 
    personalised_teacher_logits, 
    target_ids, 
    lambda_h=0.7, 
    lambda_p=0.3
)