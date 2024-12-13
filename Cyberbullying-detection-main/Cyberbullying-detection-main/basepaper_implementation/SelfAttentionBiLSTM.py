import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionBiLSTM(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionBiLSTM, self).__init__()
        
        self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=input_dim, bidirectional=True, batch_first=True)
        self.fc_q = nn.Linear(1536, 512)
        self.fc_k = nn.Linear(1536, 512)
        self.fc_v = nn.Linear(1536, 512)
        
    def attention_fusion(self, vec1, vec2):
        img_text = torch.cat((vec1, vec2), 1)
        prob_img = torch.sigmoid(self.prob_img(img_text))
        prob_txt = torch.sigmoid(self.prob_text(img_text))
        
        vec1 = prob_img * vec1
        vec2 = prob_txt * vec2
        
        out_rep = torch.cat((vec1, vec2), 1)
        
        return out_rep 
    
    def forward(self, x):
        # Ensure input has batch, sequence, and feature dimensions
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add a sequence length dimension for single-time-step inputs
        #print(f"Input Shape after unsqueeze: {x.shape}")

        batch_size = x.shape[0]
        #print(f"Batch Size: {batch_size}")
        
        # Initialize h0 and c0 for LSTM with batch size
        h0 = torch.randn(2 * 1, batch_size, 768).to(x.device)  # (num_layers * num_directions, batch_size, hidden_size)
        c0 = torch.randn(2 * 1, batch_size, 768).to(x.device)  # (num_layers * num_directions, batch_size, hidden_size)
                
        after_lstm = self.bilstm(x, (h0, c0))[0]

        q = F.relu(self.fc_q(after_lstm))
        k = F.relu(self.fc_k(after_lstm))
        v = F.relu(self.fc_v(after_lstm))

        att = F.tanh(torch.bmm(q, k.transpose(1, 2)))  # Transpose k for batch matmul
        #print("Attention Shape :", att.shape)
        
        soft = F.softmax(att, dim=-1)
        #print("Softmax Shape :", soft.shape)
        
        value = torch.mean(torch.bmm(soft, v), dim=1)
        #print("Value Shape :", value.shape)
        
        return value