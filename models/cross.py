import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_attention_a import MultiheadAttention


class TransformerEncoderLayerPreNorm(nn.Module):

    def __init__(self, d_model,  dim_feedforward ,nhead, dropout, activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.activation = nn.ReLU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, center_features, grouped_features, tgt_mask= None,
                memory_mask= None, tgt_key_padding_mask= None,
                memory_key_padding_mask=None):
        src2 = self.self_attn(center_features, grouped_features, grouped_features)[0]
        src = center_features + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src2 =self.activation(self.linear1(src))
        # src2 =self.linear2(self.dropout(src2))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats//2, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats//2, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)     ##(b,c,np,ns)
        return position_embedding


class Att(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self,cha):
        super().__init__()
        self.decoder_attention = nn.TransformerDecoder(
            TransformerEncoderLayerPreNorm(d_model=cha, dim_feedforward=cha * 2, nhead=4, dropout=0.1),
            num_layers=4)
        self.decoder_pe = PositionEmbeddingLearned(3, cha)
        self.decoder_pe_new = PositionEmbeddingLearned(3, cha)

    def forward(self, xyz,feature,xyz_new,feature_new):
        #xyz (b,n,3)
        #feature (b,c,n)
        xyz = xyz.transpose(1, 2).contiguous()  #(b,3,n)
        position_encoding = self.decoder_pe(xyz)  #(b,c,n)
        feature = (feature+position_encoding).transpose(1, 2).contiguous()   #(b,n1,c)

        xyz_new = xyz_new.transpose(1, 2).contiguous()  # (b,3,n1)
        position_encodingnew = self.decoder_pe_new(xyz_new)  # (b,c,n)
        feature_new = (feature_new + position_encodingnew).transpose(1, 2).contiguous()  #(b,n2,c)

        x = self.decoder_attention(feature_new,feature)
        x = x.transpose(1, 2).contiguous()
        return x