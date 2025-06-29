# Graph showing Decoder Options
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchcrf import CRF

from method_1 import (
    normalise_signal,
    collate_fn_seq,
    PositionalEncoding,
    Seq2SeqMotifCaller,
    evaluate_model_seq
)

from method_1 import Decoder as Decoder_GRU

from method_2 import (
    MotifSeqDataset,
    train_model
)

from method_3 import (
    Encoder_CNNTransformer,
    Encoder_CNN_BiGRU
)

# -------------------------
# Global Tokens and Settings
# -------------------------
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2


# -------------------------
# Encoder Variants
# -------------------------
# (1) CNN+Transformer Encoder (CNN backbone + Transformer layers)

# (2) CNN+BiGRU Encoder (CNN backbone + bidirectional GRU)



#######################################
# Decoder Variants
#######################################
# (1) GRU Decoder with Attention

    
# (2) LSTM Decoder with Attention
class Decoder_LSTM(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_size, num_layers=1, dropout=0.1):
        super(Decoder_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.lstm = nn.LSTM(
            d_model + d_model,       # input is concatenated embedding + context vector
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attn = nn.Linear(hidden_size + d_model, 1)  # attention score over encoder outputs
        self.out = nn.Linear(hidden_size, vocab_size)    # project hidden state to vocab logits
        self.hidden_size = hidden_size
        
    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.5, max_length=100):
        # encoder_outputs: (T_enc, B, d_model)
        T_enc, batch_size, d_model = encoder_outputs.size()
        # convert to (B, T_enc, d_model) for easier batch ops
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        device = encoder_outputs.device

        # initialise LSTM hidden and cell states to zeros
        hidden = (
            torch.zeros(1, batch_size, self.hidden_size, device=device),
            torch.zeros(1, batch_size, self.hidden_size, device=device)
        )

        # start with SOS token for each sample
        input_token = torch.full((batch_size, 1), SOS_TOKEN, dtype=torch.long, device=device)
        outputs = []

        for t in range(max_length):
            # embed the current input token
            embedded = self.embedding(input_token)  # (B, 1, d_model)

            # compute attention weights
            hidden_last = hidden[0][-1].unsqueeze(1)           # (B,1,hidden)
            hidden_repeat = hidden_last.repeat(1, T_enc, 1)    # (B,T_enc,hidden)
            attn_input = torch.cat((hidden_repeat, encoder_outputs), dim=2)
            attn_weights = F.softmax(self.attn(attn_input).squeeze(2), dim=1)  
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (B,1,d_model)

            # concatenate embedding and context, pass through LSTM
            lstm_input = torch.cat((embedded, context), dim=2)  # (B,1,d_model*2)
            output, hidden = self.lstm(lstm_input, hidden)     # output (B,1,hidden)

            # project to vocabulary and store
            prediction = self.out(output.squeeze(1))           # (B,vocab)
            outputs.append(prediction.unsqueeze(1))             # collect for each time step

            # decide next input via teacher forcing or greedy
            if targets is not None and t < targets.size(1):
                if torch.rand(1).item() < teacher_forcing_ratio:
                    input_token = targets[:, t].unsqueeze(1)
                else:
                    input_token = prediction.argmax(dim=1, keepdim=True)
            else:
                input_token = prediction.argmax(dim=1, keepdim=True)

        # concatenate all time steps: (B, max_length, vocab)
        return torch.cat(outputs, dim=1)


# (3) Vanilla RNN Decoder with Attention
class Decoder_RNN(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_size, num_layers=1, dropout=0.1):
        super(Decoder_RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.rnn = nn.RNN(
            d_model + d_model,       # input embedding + context
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
            dropout=dropout
        )
        self.attn = nn.Linear(hidden_size + d_model, 1)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        
    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.5, max_length=100):
        # encoder_outputs: (T_enc, B, d_model) -> (B, T_enc, d_model)
        T_enc, batch_size, d_model = encoder_outputs.size()
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        device = encoder_outputs.device

        # initialise hidden state
        hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)
        input_token = torch.full((batch_size, 1), SOS_TOKEN, dtype=torch.long, device=device)
        outputs = []

        for t in range(max_length):
            embedded = self.embedding(input_token)  # (B,1,d_model)

            # attention over encoder outputs
            hidden_last = hidden[-1].unsqueeze(1)
            hidden_repeat = hidden_last.repeat(1, T_enc, 1)
            attn_input = torch.cat((hidden_repeat, encoder_outputs), dim=2)
            attn_weights = F.softmax(self.attn(attn_input).squeeze(2), dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

            # RNN step
            rnn_input = torch.cat((embedded, context), dim=2)
            output, hidden = self.rnn(rnn_input, hidden)

            prediction = self.out(output.squeeze(1))
            outputs.append(prediction.unsqueeze(1))

            # next token choice
            if targets is not None and t < targets.size(1):
                if torch.rand(1).item() < teacher_forcing_ratio:
                    input_token = targets[:, t].unsqueeze(1)
                else:
                    input_token = prediction.argmax(dim=1, keepdim=True)
            else:
                input_token = prediction.argmax(dim=1, keepdim=True)

        return torch.cat(outputs, dim=1)


# (4) Transformer Decoder
class Decoder_Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers=2, nhead=4, dropout=0.1):
        super(Decoder_Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        self.out = nn.Linear(d_model, vocab_size)
        
    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.5, max_length=100):
        # encoder_outputs: (T_enc, B, d_model)
        device = encoder_outputs.device
        outputs = []

        # start with SOS token
        input_token = torch.full((encoder_outputs.size(1), 1), SOS_TOKEN, dtype=torch.long, device=device)
        tgt_seq = self.embedding(input_token).transpose(0,1)  # (1, B, d_model)
        tgt_seq = self.pos_enc(tgt_seq)

        for t in range(max_length):
            # generate subsequent mask for autoregressive decoding
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(t+1).to(device)
            dec_output = self.transformer_decoder(
                tgt_seq,
                encoder_outputs,
                tgt_mask=tgt_mask
            )
            # take last time step and project
            out_t = self.out(dec_output[-1])  # (B, vocab)
            outputs.append(out_t.unsqueeze(1))

            # choose next token
            next_token = out_t.argmax(dim=1, keepdim=True)
            if targets is not None and t < targets.size(1):
                if torch.rand(1).item() < teacher_forcing_ratio:
                    input_token = targets[:, t].unsqueeze(1)
                else:
                    input_token = next_token
            else:
                input_token = next_token

            # append new embedding
            new_emb = self.embedding(input_token).transpose(0,1)
            new_emb = self.pos_enc(new_emb)
            tgt_seq = torch.cat((tgt_seq, new_emb), dim=0)

        return torch.cat(outputs, dim=1)


# (5) Attention-only Decoder
class Decoder_AttentionOnly(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1):
        super(Decoder_AttentionOnly, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.query = nn.Parameter(torch.zeros(1, d_model))  # learned global query
        self.attn = nn.Linear(d_model + d_model, 1)
        self.out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.5, max_length=100):
        # encoder_outputs: (T_enc, B, d_model) -> (B, T_enc, d_model)
        T_enc, batch_size, d_model = encoder_outputs.size()
        enc_out = encoder_outputs.permute(1, 0, 2)
        device = enc_out.device
        query = self.query.expand(batch_size, -1)
        outputs = []

        for _ in range(max_length):
            # compute attention weights over encoder outputs using current query
            query_exp = query.unsqueeze(1).expand(-1, T_enc, -1)
            attn_input = torch.cat((query_exp, enc_out), dim=2)
            attn_weights = F.softmax(self.attn(attn_input).squeeze(2), dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_out).squeeze(1)

            # project context directly to vocab
            output = self.out(self.dropout(context))
            outputs.append(output.unsqueeze(1))

            # update query embedding for next step
            next_token = output.argmax(dim=1, keepdim=True)
            query = 0.5 * query + 0.5 * self.embedding(next_token).squeeze(1)

        return torch.cat(outputs, dim=1)


# (6) Hybrid Decoder (GRU + LSTM)
class Decoder_Hybrid(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_size, num_layers=1, dropout=0.1):
        super(Decoder_Hybrid, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU( d_model + d_model, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0 )
        self.lstm = nn.LSTM(d_model + d_model, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attn = nn.Linear(hidden_size + d_model, 1)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        
    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.5, max_length=100):
        # encoder_outputs: (T_enc, B, d_model) -> (B, T_enc, d_model)
        enc = encoder_outputs.permute(1, 0, 2)
        device = enc.device
        hidden_gru = torch.zeros(1, enc.size(0), self.hidden_size, device=device)
        hidden_lstm = (
            torch.zeros(1, enc.size(0), self.hidden_size, device=device),
            torch.zeros(1, enc.size(0), self.hidden_size, device=device)
        )
        input_token = torch.full((enc.size(0), 1), SOS_TOKEN, dtype=torch.long, device=device)
        outputs = []

        for t in range(max_length):
            embedded = self.embedding(input_token)
            last_gru = hidden_gru[-1].unsqueeze(1).repeat(1, enc.size(1), 1)
            attn_input = torch.cat((last_gru, enc), dim=2)
            attn_weights = F.softmax(self.attn(attn_input).squeeze(2), dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), enc)

            # run both GRU and LSTM on same input+context
            rnn_input = torch.cat((embedded, context), dim=2)
            out_gru, hidden_gru = self.gru(rnn_input, hidden_gru)
            out_lstm, hidden_lstm = self.lstm(rnn_input, hidden_lstm)

            # average their outputs
            combined = (out_gru + out_lstm) / 2.0
            prediction = self.out(combined.squeeze(1))
            outputs.append(prediction.unsqueeze(1))

            # next input
            if targets is not None and t < targets.size(1) and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = targets[:, t].unsqueeze(1)
            else:
                input_token = prediction.argmax(dim=1, keepdim=True)

        return torch.cat(outputs, dim=1)


# (7) Conformer Decoder (Simplified)
class Decoder_Conformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers=2, nhead=4, dropout=0.1):
        super(Decoder_Conformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)
        
    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.5, max_length=100):
        # encoder_outputs: (T_enc, B, d_model)
        device = encoder_outputs.device
        outputs = []
        input_token = torch.full((encoder_outputs.size(1), 1), SOS_TOKEN, dtype=torch.long, device=device)
        tgt_seq = self.embedding(input_token).transpose(0,1)  # (1, B, d_model)
        tgt_seq = self.pos_enc(tgt_seq)

        for t in range(max_length):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(t+1).to(device)
            dec_output = self.transformer_decoder(tgt_seq, encoder_outputs, tgt_mask=tgt_mask)

            # apply convolutional block on decoder output
            conv_in = dec_output.transpose(0,1).transpose(1,2)
            conv_out = self.conv_block(conv_in).mean(dim=2)
            prediction = self.out(conv_out)  # (B, vocab)
            outputs.append(prediction.unsqueeze(1))

            # next token
            next_token = prediction.argmax(dim=1, keepdim=True)
            if targets is not None and t < targets.size(1) and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = targets[:, t].unsqueeze(1)
            else:
                input_token = next_token

            new_emb = self.embedding(input_token).transpose(0,1)
            new_emb = self.pos_enc(new_emb)
            tgt_seq = torch.cat((tgt_seq, new_emb), dim=0)

        return torch.cat(outputs, dim=1)


# (8) GRU Decoder without Attention
class Decoder_GRU_NoAttn(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_size, num_layers=1, dropout=0.1):
        super(Decoder_GRU_NoAttn, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(
            d_model,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.out = nn.Linear(hidden_size, vocab_size)
        self.fc_init = nn.Linear(d_model, hidden_size)  # map encoder final vector to hidden_size

    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.5, max_length=100):
        # encoder_outputs: (T_enc, B, d_model)
        context = encoder_outputs[-1]  # final time step as summary (B, d_model)
        hidden = self.fc_init(context).unsqueeze(0)  # initial hidden (1, B, hidden_size)
        input_token = torch.full((context.size(0), 1), SOS_TOKEN, dtype=torch.long, device=context.device)
        outputs = []

        for t in range(max_length):
            embedded = self.embedding(input_token)         # (B,1,d_model)
            out, hidden = self.gru(embedded, hidden)       # (B,1,hidden)
            prediction = self.out(out.squeeze(1))          # (B,vocab)
            outputs.append(prediction.unsqueeze(1))

            if targets is not None and t < targets.size(1) and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = targets[:, t].unsqueeze(1)
            else:
                input_token = prediction.argmax(dim=1, keepdim=True)

        return torch.cat(outputs, dim=1)


# (9) LSTM Decoder without Attention
class Decoder_LSTM_NoAttn(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_size, num_layers=1, dropout=0.1):
        super(Decoder_LSTM_NoAttn, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.lstm = nn.LSTM(
            d_model,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.out = nn.Linear(hidden_size, vocab_size)
        self.fc_init = nn.Linear(d_model, hidden_size)

    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.5, max_length=100):
        # encoder_outputs: (T_enc, B, d_model)
        context = encoder_outputs[-1]  # (B, d_model)
        # initialise hidden and cell from context
        hidden = (
            self.fc_init(context).unsqueeze(0),
            self.fc_init(context).unsqueeze(0)
        )
        input_token = torch.full((context.size(0), 1), SOS_TOKEN, dtype=torch.long, device=context.device)
        outputs = []

        for t in range(max_length):
            embedded = self.embedding(input_token)
            out, hidden = self.lstm(embedded, hidden)
            prediction = self.out(out.squeeze(1))
            outputs.append(prediction.unsqueeze(1))

            if targets is not None and t < targets.size(1) and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = targets[:, t].unsqueeze(1)
            else:
                input_token = prediction.argmax(dim=1, keepdim=True)

        return torch.cat(outputs, dim=1)


# (10) CTC Decoder
class Decoder_CTC(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1):
        super(Decoder_CTC, self).__init__()
        # direct linear projection from encoder features to vocab logits
        self.out = nn.Linear(d_model, vocab_size)
    
    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.0, max_length=None):
        # encoder_outputs: (T_enc, B, d_model)
        logits = self.out(encoder_outputs)        # (T_enc, B, vocab)
        return logits.transpose(0,1)              # (B, T_enc, vocab)


# (11) CRF Decoder
class Decoder_CRF(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1):
        super().__init__()
        self.hidden2tag = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(vocab_size, batch_first=True)

    def forward(self, encoder_outputs, targets=None, mask=None, **_ignored):
        # encoder_outputs: (T_enc, B, d_model)
        emissions = self.hidden2tag(self.dropout(encoder_outputs)).permute(1, 0, 2)
        # if targets provided, return negative log-likelihood for training
        if targets is not None:
            return -self.crf(emissions, targets, mask=mask)
        # otherwise decode best tag sequence for inference
        best_paths = self.crf.decode(emissions, mask=mask)
        max_len = emissions.size(1)
        # pad decoded paths to max_len
        padded = [seq + [PAD_TOKEN]*(max_len-len(seq)) for seq in best_paths]
        return torch.tensor(padded, device=emissions.device)  # (B, T_enc)


# Greedy decoding helper adapted for CTC/CRF outputs
def greedy_decode_seq(outputs):
    # if 3-dim (logits), take argmax; if 2-dim (CRF tags), use directly
    if outputs.dim() == 3:
        preds = outputs.argmax(dim=2)
    elif outputs.dim() == 2:
        preds = outputs
    else:
        raise ValueError(f"Unexpected output shape: {tuple(outputs.shape)}")

    decoded = []
    for seq in preds.tolist():
        tokens = []
        for token in seq:
            if token == EOS_TOKEN:
                break
            if token in (SOS_TOKEN, PAD_TOKEN):
                continue
            tokens.append(token)
        decoded.append(tokens)
    return decoded


# Training loop supporting both CE and CRF decoders
def train_model(model,
                train_loader,
                val_loader,
                device,
                num_epochs=30,
                lr=1e-3,
                weight_decay=1e-4,
                grad_clip=1.0,
                max_length=100,
                teacher_forcing_init=0.5,
                teacher_forcing_final=0.0,
                decay_rate=50,
                patience=5,
                save_dir="graphs"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience, verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    is_crf = hasattr(model.decoder, "crf")

    train_losses, val_losses = [], []
    token_accs, seq_accs = [], []
    best_val_loss, epochs_no_improve = float("inf"), 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        # exponential decay of teacher forcing ratio
        tf_ratio = teacher_forcing_final + (teacher_forcing_init - teacher_forcing_final) * np.exp(-epoch / decay_rate)
        print(f"Epoch {epoch}: Teacher Forcing Ratio = {tf_ratio:.3f}")

        # TRAINING
        for signals, targets, _ in train_loader:
            signals, targets = signals.to(device), targets.to(device)
            optimizer.zero_grad()

            if is_crf:
                # prepare inputs for CRF loss
                enc_out = model.encoder(signals)
                T_enc, B = enc_out.size(0), enc_out.size(1)
                gold = targets[:, 1:]  # drop SOS
                gold_lens = (gold != PAD_TOKEN).sum(dim=1)
                # pad or trim to T_enc
                if gold.size(1) < T_enc:
                    pad = torch.full((B, T_enc - gold.size(1)), PAD_TOKEN, dtype=gold.dtype, device=device)
                    gold = torch.cat([gold, pad], dim=1)
                else:
                    gold = gold[:, :T_enc]
                idx = torch.arange(T_enc, device=device).expand(B, -1)
                mask = idx < gold_lens.unsqueeze(1)
                loss = model.decoder(enc_out, targets=gold, mask=mask)
            else:
                # standard CE training
                outputs = model(signals, targets=targets[:, :-1], teacher_forcing_ratio=tf_ratio, max_length=targets.size(1)-1)
                min_len = min(outputs.size(1), targets.size(1)-1)
                logits = outputs[:, :min_len, :].reshape(-1, model.decoder.out.out_features)
                gold   = targets[:, 1:min_len+1].reshape(-1)
                loss   = criterion(logits, gold)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # VALIDATION
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for signals, targets, _ in val_loader:
                signals, targets = signals.to(device), targets.to(device)

                if is_crf:
                    enc_out = model.encoder(signals)
                    T_enc, B = enc_out.size(0), enc_out.size(1)
                    gold = targets[:, 1:]
                    gold_lens = (gold != PAD_TOKEN).sum(dim=1)
                    if gold.size(1) < T_enc:
                        pad = torch.full((B, T_enc - gold.size(1)), PAD_TOKEN, dtype=gold.dtype, device=device)
                        gold = torch.cat([gold, pad], dim=1)
                    else:
                        gold = gold[:, :T_enc]
                    idx = torch.arange(T_enc, device=device).expand(B, -1)
                    mask = idx < gold_lens.unsqueeze(1)
                    loss = model.decoder(enc_out, targets=gold, mask=mask)
                else:
                    outputs = model(signals, targets=None, teacher_forcing_ratio=0.0, max_length=targets.size(1)-1)
                    min_len = min(outputs.size(1), targets.size(1)-1)
                    logits  = outputs[:, :min_len, :].reshape(-1, model.decoder.out.out_features)
                    gold    = targets[:, 1:min_len+1].reshape(-1)
                    loss    = criterion(logits, gold)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # compute accuracy only for non-CRF
        if not is_crf:
            token_acc, seq_acc, _, _ = evaluate_model_seq(model, val_loader, device, max_length=targets.size(1)-1)
        else:
            token_acc = seq_acc = 0.0

        token_accs.append(token_acc)
        seq_accs.append(seq_acc)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        if not is_crf:
            print(f"           Token Acc = {token_acc*100:.2f}%, Seq Acc = {seq_acc*100:.2f}%")

        # EARLY STOPPING
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    # PLOT LOSS AND ACCURACY CURVES
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train")
    plt.plot(range(1, len(val_losses)+1), val_losses,   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"))
    plt.close()

    if not is_crf:
        plt.figure()
        plt.plot(range(1, len(token_accs)+1), token_accs, label="Token")
        plt.plot(range(1, len(seq_accs)+1),   seq_accs,   label="Seq")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "accuracy_curves.png"))
        plt.close()

    return model, best_val_loss






# -------------------------
# Main Execution: Comparing Encoder Architectures
# -------------------------
if __name__ == "__main__":
    # Define all encoder variants to compare.
    # Four CNN-based variants plus pure approaches.
    encoder_variants = {
        "CNN_Transformer": Encoder_CNNTransformer,
        "CNN_BiGRU": Encoder_CNN_BiGRU
    }
    
    decoder_variants = {
        "GRU": Decoder_GRU,
        "LSTM": Decoder_LSTM,
        "RNN": Decoder_RNN,
        "Transformer": Decoder_Transformer,
        "AttentionOnly": Decoder_AttentionOnly,
        "Hybrid": Decoder_Hybrid,
        "Conformer": Decoder_Conformer,
        "GRU_NoAttn": Decoder_GRU_NoAttn,
        "LSTM_NoAttn": Decoder_LSTM_NoAttn,
        "CTC": Decoder_CTC,
        "CRF": Decoder_CRF
    }

    # File path to the CSV file.
    csv_path = os.path.join("squigulator", "longer_large_simplified_results.csv")
    # Use a subset (e.g. 50,000 rows) for quicker experiments.
    sample_size = 50000

    # Hyperparameters (fixed for comparison)
    final_d_model = 256
    final_nhead = 4          # Only used for Transformer-based variants.
    final_num_layers = 2     # For Transformer, BiLSTM, and RNN.
    final_dropout = 0.1
    final_hidden_size = 128
    teacher_forcing_init = 0.5
    teacher_forcing_final = 0.0
    learning_rate = 1e-3
    weight_decay = 1e-4
    batch_size = 16
    num_epochs = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[INFO] Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Using CPU.")

    results = {}

    for enc_name, encoder_class in encoder_variants.items():
        print(f"\n=== Running experiment with encoder variant: {enc_name} ===")
        results[enc_name] = {}
        for dec_name, decoder_class in decoder_variants.items():
            print(f"\n=== Decoder variant: {dec_name} ===")
            # Load dataset (using robust normalisation).
            dataset = MotifSeqDataset(csv_path, norm_method="robust_no_centre", debug=True, sample_size=sample_size)
            total_samples = len(dataset)
            indices = list(range(total_samples))
            train_size = int(0.6 * total_samples)
            val_size = int(0.2 * total_samples)
            train_indices, rest_indices = train_test_split(indices, train_size=train_size, random_state=42)
            val_indices, test_indices = train_test_split(rest_indices, train_size=val_size, random_state=42)
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            test_dataset = Subset(dataset, test_indices)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_seq)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_seq)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_seq)

            if enc_name == "CNN_Transformer":
                encoder = encoder_class(input_channels=1, d_model=final_d_model, nhead=final_nhead,
                                          num_layers=final_num_layers, dropout=final_dropout)
            elif enc_name == "CNN_BiGRU":
                encoder = encoder_class(input_channels=1, d_model=final_d_model, num_layers=final_num_layers, dropout=final_dropout)
            else:
                raise ValueError("Unknown encoder variant")
            
            
            if dec_name in ["GRU", "LSTM", "RNN", "Hybrid"]:
                decoder = decoder_class(vocab_size=dataset.vocab_size, d_model=final_d_model,
                                          hidden_size=final_hidden_size, num_layers=1, dropout=final_dropout)
            elif dec_name in ["Transformer", "Conformer"]:
                decoder = decoder_class(vocab_size=dataset.vocab_size, d_model=final_d_model, num_layers=final_num_layers,
                                          nhead=final_nhead, dropout=final_dropout)
            elif dec_name == "AttentionOnly":
                decoder = decoder_class(vocab_size=dataset.vocab_size, d_model=final_d_model, dropout=final_dropout)
            elif dec_name in ["GRU_NoAttn", "LSTM_NoAttn"]:
                decoder = decoder_class(vocab_size=dataset.vocab_size, d_model=final_d_model,
                                          hidden_size=final_hidden_size, num_layers=1, dropout=final_dropout)
            elif dec_name in ["CTC", "CRF"]:
                decoder = decoder_class(vocab_size=dataset.vocab_size, d_model=final_d_model, dropout=final_dropout)
            else:
                raise ValueError("Unknown decoder variant.")
            
            model = Seq2SeqMotifCaller(encoder, decoder)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.to(device)

            # For demonstration, here we use test_loader for both training and validation.
            save_dir = f"graphs_{enc_name}_{dec_name}"
            os.makedirs(save_dir, exist_ok=True)
            # Train the model.
            model, best_val_loss = train_model(model, train_loader, val_loader, device, num_epochs=num_epochs,
                                            lr=learning_rate, weight_decay=weight_decay, max_length=20,
                                            teacher_forcing_init=teacher_forcing_init, teacher_forcing_final=teacher_forcing_final,
                                            save_dir=save_dir)

            token_acc, seq_acc, _, _ = evaluate_model_seq(model, test_loader, device, dataset.idx2motif, max_length=20)
            # avg_time, peak_mem = evaluate_inference(model, dummy_batch, n_iters=50, device=device)
            results[enc_name][dec_name] = {
                "token_acc": token_acc,
                "seq_acc": seq_acc
                # "inference_time": avg_time,f
                # "peak_memory": peak_mem if peak_mem is not None else 0.0
            }
            print(f"[{enc_name} + {dec_name}] Token Acc: {token_acc*100:.2f}%, Seq Acc: {seq_acc*100:.2f}%")
            # if device.type == "cuda":
            #     print(f"Inference Time: {avg_time:.2f} ms; Peak Memory: {peak_mem:.2f} MB")
            # else:
            #     print("Memory usage not measured on CPU.")
    
    #######################################
    # Plot Bar Charts for Metrics per Encoder Backbone
    #######################################
    for enc_name in encoder_variants.keys():
        dec_names = list(results[enc_name].keys())
        token_accuracies = [results[enc_name][d]["token_acc"] * 100 for d in dec_names]
        seq_accuracies = [results[enc_name][d]["seq_acc"] * 100 for d in dec_names]
        # inf_times = [results[enc_name][d]["inference_time"] for d in dec_names]
        # mem_usages = [results[enc_name][d]["peak_memory"] for d in dec_names]
        x = np.arange(len(dec_names))
        width = 0.35
        
        # Accuracy Comparison
        plt.figure(figsize=(8,4))
        plt.bar(x - width/2, token_accuracies, width, label='Token Acc', color='skyblue')
        plt.bar(x + width/2, seq_accuracies, width, label='Seq Acc', color='lightgreen')
        plt.ylabel('Accuracy (%)', fontsize=10)
        plt.xlabel('Decoder Variant', fontsize=10)
        plt.title(f'{enc_name}: Accuracy Comparison', fontsize=12)
        plt.xticks(x, dec_names, rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        os.makedirs("summary_graphs", exist_ok=True)
        plt.savefig(os.path.join("summary_graphs", f"{enc_name}_accuracy_comparison.png"), dpi=150)
        plt.close()
        
        # # Inference Time Comparison
        # plt.figure(figsize=(8,4))
        # plt.bar(x, inf_times, width, color='skyblue')
        # plt.ylabel('Inference Time (ms)', fontsize=10)
        # plt.xlabel('Decoder Variant', fontsize=10)
        # plt.title(f'{enc_name}: Inference Time', fontsize=12)
        # plt.xticks(x, dec_names, rotation=45, fontsize=8)
        # plt.yticks(fontsize=8)
        # plt.tight_layout()
        # plt.savefig(os.path.join("summary_graphs", f"{enc_name}_inference_time.png"), dpi=150)
        # plt.close()
        
        # # Memory Usage Comparison (if CUDA is used)
        # if device.type == "cuda":
        #     plt.figure(figsize=(8,4))
        #     plt.bar(x, mem_usages, width, color='salmon')
        #     plt.ylabel('Peak Memory (MB)', fontsize=10)
        #     plt.xlabel('Decoder Variant', fontsize=10)
        #     plt.title(f'{enc_name}: GPU Memory Usage', fontsize=12)
        #     plt.xticks(x, dec_names, rotation=45, fontsize=8)
        #     plt.yticks(fontsize=8)
        #     plt.tight_layout()
        #     plt.savefig(os.path.join("summary_graphs", f"{enc_name}_memory_usage.png"), dpi=150)
        #     plt.close()

    #######################################
    # Print Final Summary
    #######################################
    print("\n=== Final Results Summary ===")
    for enc_name, dec_data in results.items():
        print(f"\nEncoder Backbone: {enc_name}")
        for dec_name, metrics in dec_data.items():
            # mem_str = f"{metrics['peak_memory']:.2f} MB" if device.type=="cuda" else "N/A"
            print(f"  {dec_name}: Token Acc = {metrics['token_acc']*100:.2f}%, Seq Acc = {metrics['seq_acc']*100:.2f}% ")  # Inference Time = {metrics['inference_time']:.2f} ms, Peak Memory = {mem_str}")