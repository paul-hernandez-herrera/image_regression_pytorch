import torch

class MDE_points(torch.nn.Module):
    def __init__(self, reduction = 'sum'):
        self.reduction = reduction
        super().__init__()

    def forward(self, model_output, target_points):
        # this functions assumes [B, rows, columns]. This function assumes we have B target outputs, with rows indicanting the number of points and columns indicating the dimensión of points
        loss = torch.mean(torch.sqrt( torch.sum(torch.square(model_output-target_points), dim = 2) ), dim=1)
        
        if self.reduction == 'sum':
            # loss equal to sum of batch
            loss = torch.sum(loss)
        return loss