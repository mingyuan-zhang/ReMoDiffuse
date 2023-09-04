import torch
import torch.nn as nn
import numpy as np
import time
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from ..builder import SUBMODULES

from mogen.models.utils.word_vectorizer import WordVectorizer


def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            

def reparameterize(mu, logvar):
    s_var = logvar.mul(0.5).exp_()
    eps = s_var.data.new(s_var.size()).normal_()
    return eps.mul(s_var).add_(mu)


# batch_size, dimension and position
# output: (batch_size, dim)
def positional_encoding(batch_size, dim, pos):
    assert batch_size == pos.shape[0]
    positions_enc = np.array([
        [pos[j] / np.power(10000, (i-i%2)/dim) for i in range(dim)]
        for j in range(batch_size)
    ], dtype=np.float32)
    positions_enc[:, 0::2] = np.sin(positions_enc[:, 0::2])
    positions_enc[:, 1::2] = np.cos(positions_enc[:, 1::2])
    return torch.from_numpy(positions_enc).float()


def get_padding_mask(batch_size, seq_len, cap_lens):
    cap_lens = cap_lens.data.tolist()
    mask_2d = torch.ones((batch_size, seq_len, seq_len), dtype=torch.float32)
    for i, cap_len in enumerate(cap_lens):
        mask_2d[i, :, :cap_len] = 0
    return mask_2d.bool(), 1 - mask_2d[:, :, 0].clone()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        return self.pe[pos]


@SUBMODULES.register_module()
class T2MMotionEncoder(nn.Module):
    
    def __init__(self,
                 input_size,
                 movement_hidden_size,
                 movement_latent_size,
                 motion_hidden_size,
                 motion_latent_size):
        super().__init__()
        self.movement_encoder = MovementConvEncoder(
            input_size=input_size-4,
            hidden_size=movement_hidden_size,
            output_size=movement_latent_size)
        self.motion_encoder = MotionEncoderBiGRUCo(
            input_size=movement_latent_size,
            hidden_size=motion_hidden_size,
            output_size=motion_latent_size
        )
        
    def load_pretrained(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.movement_encoder.load_state_dict(checkpoint['movement_encoder'])
        self.motion_encoder.load_state_dict(checkpoint['motion_encoder'])
        
    def forward(self, motion, motion_length, motion_mask):
        motion = motion.detach().float()
        sort_idx = np.argsort(motion_length.data.tolist())[::-1].copy()
        rank_idx = np.empty_like(sort_idx)
        rank_idx[sort_idx] = np.arange(len(motion_length))
        motion = motion[sort_idx]
        motion_length = motion_length[sort_idx]
        
        movements = self.movement_encoder(motion[..., :-4]).detach()
        m_lens = motion_length // 4
        motion_embedding = self.motion_encoder(movements, m_lens)
        motion_embedding_ordered = motion_embedding[rank_idx]
        return motion_embedding_ordered


@SUBMODULES.register_module()
class T2MTextEncoder(nn.Module):
    
    def __init__(self,
                 word_size,
                 pos_size,
                 hidden_size,
                 output_size,
                 max_text_len):
        super().__init__()
        self.text_encoder = TextEncoderBiGRUCo(
            word_size=word_size,
            pos_size=pos_size,
            hidden_size=hidden_size,
            output_size=output_size,
        )
        self.w_vectorizer = WordVectorizer('./data/glove', 'our_vab')
        self.max_text_len = max_text_len
        
    def load_pretrained(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        
    def forward(self, text, token, device):
        B = len(text)
        pos_one_hot = []
        word_emb = []
        sent_len = []
        for i in range(B):
            tokens = token[i].split(" ")
            if len(tokens) < self.max_text_len:
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                batch_sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - batch_sent_len)
            else:
                tokens = tokens[: self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                batch_sent_len = len(tokens)
            sent_len.append(batch_sent_len)
            batch_word_emb = []
            batch_pos_one_hot = []
            for cur_token in tokens:
                cur_word_emb, cur_pos_one_hot = self.w_vectorizer[cur_token]
                cur_word_emb = torch.from_numpy(cur_word_emb).float()
                cur_pos_one_hot = torch.from_numpy(cur_pos_one_hot).float()
                batch_word_emb.append(cur_word_emb)
                batch_pos_one_hot.append(cur_pos_one_hot)
            
            batch_word_emb = torch.stack(batch_word_emb, dim=0)
            batch_pos_one_hot = torch.stack(batch_pos_one_hot, dim=0)
            word_emb.append(batch_word_emb)
            pos_one_hot.append(batch_pos_one_hot)
        word_emb = torch.stack(word_emb, dim=0).to(device)
        pos_one_hot = torch.stack(pos_one_hot, dim=0).to(device)
        sent_len = torch.tensor(sent_len, dtype=torch.long).to(device)
        text_embedding = self.text_encoder(word_emb, pos_one_hot, sent_len)
        return text_embedding


class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size):
        super(TextEncoderBiGRUCo, self).__init__()

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True, enforce_sorted=False)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MotionEncoderBiGRUCo, self).__init__()

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)

