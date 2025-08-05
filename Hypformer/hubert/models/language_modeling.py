import torch.nn as nn

class MLMHead(nn.Module):

    def __init__(self, model):
        """
        Initialize a Masked Language Modeling head to be used for BERT-style 
        training

        Args:
            model: PyTorch model to attach MLM head to with hidden_dim and 
                   vocab_size attributes
        """
        super().__init__()
        self.linear = nn.Linear(model.hidden_dim, model.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))