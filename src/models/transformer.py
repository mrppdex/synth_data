import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

class LongitudinalDataset(Dataset):
    def __init__(self, sequences_cat, sequences_num, seq_lengths):
        self.sequences_cat = sequences_cat # List of tensors
        self.sequences_num = sequences_num # List of tensors
        self.seq_lengths = seq_lengths

    def __len__(self):
        return len(self.sequences_cat)

    def __getitem__(self, idx):
        return self.sequences_cat[idx], self.sequences_num[idx], self.seq_lengths[idx]

def collate_fn(batch):
    # Pad sequences
    cats, nums, lengths = zip(*batch)
    max_len = max(lengths)
    
    # Pad categorical with 0 (assuming 0 is padding/unknown, actual tokens start at 1)
    padded_cats = []
    for c in cats:
        pad_size = max_len - c.size(0)
        if pad_size > 0:
            padded_cats.append(torch.cat([c, torch.zeros(pad_size, c.size(1), dtype=torch.long)], dim=0))
        else:
            padded_cats.append(c)
            
    # Pad numerical with 0
    padded_nums = []
    for n in nums:
        pad_size = max_len - n.size(0)
        if pad_size > 0:
            padded_nums.append(torch.cat([n, torch.zeros(pad_size, n.size(1), dtype=torch.float)], dim=0))
        else:
            padded_nums.append(n)
            
    return torch.stack(padded_cats), torch.stack(padded_nums), torch.tensor(lengths)

class TransformerModel(nn.Module):
    def __init__(self, cat_dims, num_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings for categorical features
        # cat_dims is a list of vocab sizes for each categorical feature
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(dim, d_model) for dim in cat_dims
        ])
        
        # Projection for numerical features
        self.num_projection = nn.Linear(num_dim, d_model) if num_dim > 0 else None
        
        # We combine all features. 
        # Strategy: Sum embeddings (like BERT) or Concatenate and Project?
        # Summing requires same dimension. Concatenating increases dimension.
        # Let's Concatenate and Project to d_model.
        input_dim = len(cat_dims) * d_model + (d_model if num_dim > 0 else 0)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model)) # Simple learnable positional encoding
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.cat_heads = nn.ModuleList([
            nn.Linear(d_model, dim) for dim in cat_dims
        ])
        
        if num_dim > 0:
            self.num_head = nn.Linear(d_model, num_dim)
        else:
            self.num_head = None

    def forward(self, x_cat, x_num, src_key_padding_mask=None):
        # x_cat: (batch, seq_len, num_cat_features)
        # x_num: (batch, seq_len, num_num_features)
        
        batch_size, seq_len, _ = x_cat.size()
        
        embeddings = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            embeddings.append(emb_layer(x_cat[:, :, i]))
            
        if self.num_projection and x_num is not None:
            embeddings.append(self.num_projection(x_num))
            
        # Concatenate all embeddings
        x = torch.cat(embeddings, dim=-1)
        
        # Project to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        # Slice pos_encoder to seq_len
        if seq_len > self.pos_encoder.size(1):
             # Resize if needed (naive)
             pass 
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Create causal mask for autoregressive training
        # mask: (seq_len, seq_len) - -inf above diagonal
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        # Transformer
        output = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Heads
        cat_outputs = [head(output) for head in self.cat_heads]
        
        num_output = None
        if self.num_head:
            num_output = self.num_head(output)
            
        return cat_outputs, num_output

class CustomLongitudinalTransformer:
    def __init__(self, sequence_index, entity_columns, epochs=50, batch_size=32, constraints=None):
        self.sequence_index = sequence_index
        self.entity_columns = entity_columns
        self.epochs = epochs
        self.batch_size = batch_size
        self.constraints = constraints if constraints else {} # {col: {'min': 0, 'max': 100}}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.cat_cols = []
        self.num_cols = []
        self.int_cols = [] # Track integer columns
        self.model = None
        
    def preprocess(self, df):
        df = df.sort_values(by=self.entity_columns + [self.sequence_index])
        
        # Handle sequence index: if datetime, convert to numeric (timestamp)
        # If numeric, keep as is.
        self.is_datetime_index = False
        if pd.api.types.is_datetime64_any_dtype(df[self.sequence_index]):
            self.is_datetime_index = True
            df[self.sequence_index] = df[self.sequence_index].astype('int64') // 10**9 # Seconds
            
        # Identify columns to model (including sequence index)
        # We exclude entity columns because we generate new entities.
        cols_to_model = [c for c in df.columns if c not in self.entity_columns]
        
        # Handle OTHER datetime columns (not sequence index)
        self.datetime_cols = []
        for col in cols_to_model:
            if col != self.sequence_index and pd.api.types.is_datetime64_any_dtype(df[col]):
                self.datetime_cols.append(col)
                df[col] = df[col].astype('int64') // 10**9 # Seconds
        
        self.cat_cols = [c for c in cols_to_model if df[c].dtype == 'object' or df[c].dtype.name == 'category']
        self.num_cols = [c for c in cols_to_model if pd.api.types.is_numeric_dtype(df[c])]
        
        # Identify integer columns for post-processing enforcement
        self.int_cols = [c for c in self.num_cols if pd.api.types.is_integer_dtype(df[c])]
        
        # Encode Categorical
        df_encoded = df.copy()
        for col in self.cat_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str)) + 1 # 0 is padding
            self.label_encoders[col] = le
            
        # Scale Numerical
        if self.num_cols:
            df_encoded[self.num_cols] = self.scaler.fit_transform(df[self.num_cols].fillna(0))
            
        # Group by entity to create sequences
        grouped = df_encoded.groupby(self.entity_columns)
        
        sequences_cat = []
        sequences_num = []
        seq_lengths = []
        
        for _, group in grouped:
            if len(group) < 2:
                continue # Skip sequences too short for autoregressive training
                
            cat_seq = torch.tensor(group[self.cat_cols].values, dtype=torch.long)
            num_seq = torch.tensor(group[self.num_cols].values, dtype=torch.float)
            sequences_cat.append(cat_seq)
            sequences_num.append(num_seq)
            seq_lengths.append(len(group))
            
        return sequences_cat, sequences_num, seq_lengths

    def train(self, df, progress_callback=None):
        sequences_cat, sequences_num, seq_lengths = self.preprocess(df)
        
        if not sequences_cat:
            raise ValueError("No valid sequences found (length >= 2). Check your data.")
            
        dataset = LongitudinalDataset(sequences_cat, sequences_num, seq_lengths)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        
        # Model Config
        cat_dims = [len(self.label_encoders[col].classes_) + 1 for col in self.cat_cols] # +1 for padding
        num_dim = len(self.num_cols)
        
        self.model = TransformerModel(cat_dims, num_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        criterion_cat = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding
        criterion_num = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_cat, batch_num, lengths in dataloader:
                batch_cat = batch_cat.to(self.device)
                batch_num = batch_num.to(self.device)
                
                # Create padding mask (batch, seq_len) - True where padded
                # lengths is tensor of actual lengths
                max_len = batch_cat.size(1)
                mask = torch.arange(max_len)[None, :] >= lengths[:, None]
                mask = mask.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                # We want to predict next step.
                # Input: x[0..t] -> Output: x[1..t+1]
                # Standard autoregressive training:
                # Input is whole sequence.
                # Output at pos t is prediction for t+1.
                # We shift targets.
                
                input_cat = batch_cat[:, :-1, :]
                input_num = batch_num[:, :-1, :]
                target_cat = batch_cat[:, 1:, :]
                target_num = batch_num[:, 1:, :]
                
                # Adjust mask for shortened sequence
                mask_input = mask[:, :-1]
                
                cat_preds, num_pred = self.model(input_cat, input_num, src_key_padding_mask=mask_input)
                
                loss = 0
                # Categorical Loss
                for i, pred in enumerate(cat_preds):
                    # pred: (batch, seq_len, vocab_size)
                    # target: (batch, seq_len)
                    loss += criterion_cat(pred.reshape(-1, pred.size(-1)), target_cat[:, :, i].reshape(-1))
                    
                # Numerical Loss
                if num_pred is not None:
                    # Mask out padding for MSE
                    # target_num: (batch, seq_len, num_dim)
                    # mask_input: (batch, seq_len)
                    # We only calculate loss where mask is False (not padded)
                    active_elements = ~mask_input.unsqueeze(-1)
                    loss += criterion_num(num_pred * active_elements, target_num * active_elements)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss/len(dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
            if progress_callback:
                progress_callback(epoch + 1, avg_loss)

    def generate(self, n_samples):
        self.model.eval()
        generated_rows = []
        
        # Sample lengths from training data (simple approach: mean length or random choice)
        # For now, fixed length or random between min/max
        avg_len = 10 # Default
        
        with torch.no_grad():
            for i in range(n_samples):
                # Generate a new Subject ID
                subj_id = f"SYNTH_{i+1:03d}"
                
                # Start with padding/SOS (zeros)
                # We need to feed at least one step to get going.
                # Ideally we should have a SOS token. 
                # Since we trained with 0 as padding, starting with 0s might work if the model learned to transition from padding to data.
                # But usually padding is ignored.
                # Let's start with a random valid token for each cat and 0 for num?
                # Or better: The model is trained to predict t+1 from 0..t.
                # If we feed 0-padding, it might predict the first real token.
                
                curr_cat = torch.zeros(1, 1, len(self.cat_cols), dtype=torch.long).to(self.device)
                curr_num = torch.zeros(1, 1, len(self.num_cols), dtype=torch.float).to(self.device)
                
                seq_cats = []
                seq_nums = []
                
                # Generate sequence
                for t in range(avg_len):
                    cat_preds, num_pred = self.model(curr_cat, curr_num)
                    
                    # Sample next categorical
                    next_cats = []
                    for j, pred in enumerate(cat_preds):
                        logits = pred[:, -1, :]
                        probs = torch.softmax(logits, dim=-1)
                        # Avoid sampling 0 (padding) if possible, unless it's the only option
                        # Mask 0?
                        probs[:, 0] = 0
                        if probs.sum() == 0:
                            probs = torch.ones_like(probs) # Fallback
                        else:
                            probs = probs / probs.sum()
                            
                        next_token = torch.multinomial(probs, 1)
                        next_cats.append(next_token)
                    
                    next_cat_tensor = torch.cat(next_cats, dim=-1).unsqueeze(1)
                    
                    # Next numerical
                    next_num_tensor = num_pred[:, -1, :].unsqueeze(1) if num_pred is not None else torch.zeros(1, 1, 0).to(self.device)
                    
                    seq_cats.append(next_cat_tensor.cpu().numpy().flatten())
                    seq_nums.append(next_num_tensor.cpu().numpy().flatten())
                    
                    # Append to current sequence
                    curr_cat = torch.cat([curr_cat, next_cat_tensor], dim=1)
                    curr_num = torch.cat([curr_num, next_num_tensor], dim=1)

                # Reconstruct Data
                # Inverse Transform
                seq_cats = np.array(seq_cats)
                seq_nums = np.array(seq_nums)
                
                row_dict = {}
                # Entity Columns (Assign same ID for whole sequence)
                # Assuming single entity column for simplicity or replicate
                for ec in self.entity_columns:
                    row_dict[ec] = [subj_id] * avg_len
                    
                # Categorical
                for idx, col in enumerate(self.cat_cols):
                    # -1 because we added 1 for padding
                    tokens = seq_cats[:, idx] - 1
                    tokens = np.clip(tokens, 0, None) # Ensure no -1
                    row_dict[col] = self.label_encoders[col].inverse_transform(tokens)
                    
                # Numerical
                if self.num_cols:
                    # Inverse scale
                    nums_inv = self.scaler.inverse_transform(seq_nums)
                    for idx, col in enumerate(self.num_cols):
                        vals = nums_inv[:, idx]
                        
                        # Apply Constraints
                        if col in self.constraints:
                            c = self.constraints[col]
                            if 'min' in c:
                                vals = np.maximum(vals, c['min'])
                            if 'max' in c:
                                vals = np.minimum(vals, c['max'])
                                
                        # Enforce Integer
                        if col in self.int_cols:
                            vals = np.round(vals).astype(int)
                            
                        row_dict[col] = vals
                        
                # Create DF for this subject
                subj_df = pd.DataFrame(row_dict)
                generated_rows.append(subj_df)
                
        final_df = pd.concat(generated_rows, ignore_index=True)
        
        # Post-process sequence index if it was datetime
        if self.is_datetime_index and self.sequence_index in final_df.columns:
            final_df[self.sequence_index] = pd.to_datetime(final_df[self.sequence_index], unit='s')
            
        # Post-process other datetime columns
        if hasattr(self, 'datetime_cols'):
            for col in self.datetime_cols:
                if col in final_df.columns:
                     final_df[col] = pd.to_datetime(final_df[col], unit='s')
            
        return final_df

