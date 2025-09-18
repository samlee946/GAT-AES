import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, T5EncoderModel
import process_data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class Attention(nn.Module):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01):
        super(Attention, self).__init__()
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev

        self.att_v = None  # Initialize in `build` or `forward` based on input size
        self.att_W = None

    def build(self, input_dim, device):
        # Initialize trainable parameters with random values
        self.att_v = nn.Parameter(torch.randn(input_dim, device=device) * self.init_stdev)  # vector
        self.att_W = nn.Parameter(torch.randn(input_dim, input_dim, device=device) * self.init_stdev)  # matrix

    def forward(self, target_rep, non_target_rep, mask=None):
        if self.att_v is None or self.att_W is None:
            # Lazily initialize weights based on input size (input_dim = target_rep.size(2))
            self.build(target_rep.size(2), device=target_rep.device)

        # Project non_target_rep with att_W (linear transformation)
        key_transformed = torch.matmul(non_target_rep, self.att_W)  # (batch_size, seq_len-1, input_dim)

        # Compute the attention scores using the dot product of target_rep and key
        # target_rep shape: (batch_size, 1, input_dim)
        # key_transformed shape: (batch_size, seq_len-1, input_dim)
        attn_scores = torch.matmul(target_rep, key_transformed.transpose(1, 2))  # (batch_size, 1, seq_len-1)

        # Optional activation on attention scores (tanh)
        if self.activation == 'tanh':
            attn_scores = torch.tanh(attn_scores)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, 1, seq_len-1)

        # Apply attention weights to the non_target_rep (value is the same as key)
        weighted_sum = torch.matmul(attn_weights, non_target_rep)  # (batch_size, 1, input_dim)

        # Attention operation: either sum or mean
        if self.op == 'attsum':
            out = weighted_sum.squeeze(1)  # (batch_size, input_dim)
        elif self.op == 'attmean':
            out = weighted_sum.squeeze(1) / mask.sum(dim=1, keepdim=True)  # Handle mask for average

        return out

class BL(nn.Module):
    def __init__(self, args, encoded_prompts):
        super().__init__()
        self.args = args
        if 't5' in args.lm_model:
            self.lm = T5EncoderModel.from_pretrained(args.lm_model, trust_remote_code=True)
        else:
            self.lm = AutoModel.from_pretrained(args.lm_model, trust_remote_code=True)
        self.emb_size = self.lm.config.hidden_size
        self.score_vector_positions = process_data.get_score_vector_positions(args)
        self.num_traits = len(self.score_vector_positions)
        self.num_features = args.essay_feats_size if args.use_essay_feats else 0
        
        self.num_emb_nodes = args.GNN_num_emb_nodes
        self.num_feat_nodes = args.GNN_num_feat_nodes

        if args.dropout > 0:
            self.dropout = nn.Dropout(args.dropout)

        # self.encoded_prompts = encoded_prompts
        self.trait_weights_for_loss = None
        if self.args.trait_weights_for_loss is not None:
            if self.args.infer_trait_weights_for_loss == 'learnable':
                self.trait_weights_for_loss = nn.Parameter(torch.tensor(args.trait_weights_for_loss, dtype=torch.float32), requires_grad=True)
            else:
                self.trait_weights_for_loss = torch.tensor(args.trait_weights_for_loss, dtype=torch.float32, requires_grad=False)

        
        # Initial transformations to hidden_size
        self.emb_transform = nn.Linear(self.emb_size, self.args.GNN_hidden_size) 
        self.feat_transform = nn.Linear(self.num_features, self.args.GNN_hidden_size) 
        
        # Trait nodes (initialized in hidden_size dimension)
        self.trait_nodes = nn.Parameter(torch.randn(self.num_traits, self.args.GNN_hidden_size))
        
        # # trait attention mechanism
        # self.trait_attention_layers = nn.ModuleList([Attention() for _ in range(self.num_traits)])

        # Separate output layers for each trait
        self.output_layers = nn.ModuleList([nn.Linear(self.args.GNN_hidden_size * 2, 1) for _ in range(self.num_traits)])
        
        # # Create static edge structure
        # self.register_buffer('edge_index', self._create_edge_structure())


    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # # Ensure that all members are moved to the specified device
        # if self.args.layer_weight_scheme == 'simple':
        #     for trait in self.score_vector_positions.keys():
        #         setattr(self, f'layer_weights_{trait}', getattr(self, f'layer_weights_{trait}').to(*args, **kwargs))
        # elif self.args.layer_weight_scheme == 'prompt_attention':
        #     for trait in self.score_vector_positions.keys():
        #         setattr(self, f'trait_query_embeddings_{trait}', getattr(self, f'trait_query_embeddings_{trait}').to(*args, **kwargs))
        #     self.encoded_prompts = {k: v.to(*args, **kwargs) for k, v in self.encoded_prompts.items()}
        if self.trait_weights_for_loss is not None:
            self.trait_weights_for_loss = self.trait_weights_for_loss.to(*args, **kwargs)
        return self
        
    def forward(self, essay_ids, prompt_ids, essay_input_ids, essay_attention_mask, essay_token_type_ids, essay_feats, labels=None):

        batch_size = essay_ids.shape[0]

        # get essay embeddings
        essay_input_ids = essay_input_ids.squeeze(1) # shape (batch_size, seq_len)
        essay_attention_mask = essay_attention_mask.squeeze(1) # shape (batch_size, seq_len)
        if essay_token_type_ids is not None:
            essay_token_type_ids = essay_token_type_ids.squeeze(1) # shape (batch_size, seq_len)
            lm_outputs_essay = self.lm(input_ids=essay_input_ids, attention_mask=essay_attention_mask, token_type_ids=essay_token_type_ids, output_hidden_states=True)
            ## essay embeddings by [CLS]
            essay_embeddings = lm_outputs_essay.last_hidden_state[:, 0, :] # shape (batch_size, hidden_size)
        else:
            lm_outputs_essay = self.lm(input_ids=essay_input_ids, attention_mask=essay_attention_mask, output_hidden_states=True)
            if 't5' in self.args.lm_model:
                # use mean pooling for T5 model as it doesn't have a [CLS] token
                # mean pooling for T5 and using attention mask to avoid padding tokens
                # sum the embeddings and divide by the number of non-padding tokens
                essay_embeddings = (lm_outputs_essay.last_hidden_state * essay_attention_mask.unsqueeze(-1)).sum(dim=1) / essay_attention_mask.sum(dim=1, keepdim=True)
            else:
                ## essay embeddings by [CLS]
                essay_embeddings = lm_outputs_essay.last_hidden_state[:, 0, :] # shape (batch_size, hidden_size)
        
        # Transform inputs
        emb_transformed = self.emb_transform(essay_embeddings)  # [batch_size, GNN_hidden_size]
        feat_transformed = self.feat_transform(essay_feats)  # [batch_size, GNN_hidden_size]

        # concat emb and feat
        x = torch.cat([emb_transformed, feat_transformed], dim=1)  # [batch_size, GNN_hidden_size * 2]
        preds = []
        for j in range(self.num_traits):
            logits = self.output_layers[j](x)  # [batch_size, 1]
            preds.append(logits)
        preds = torch.stack(preds, dim=1).squeeze(2)  # [batch_size, num_traits]
        preds = torch.sigmoid(preds)

        if labels is None:
            return None, preds

        loss = self.mask_loss_fct(preds, labels, self.trait_weights_for_loss)

        return loss, preds
    
    def mask_loss_fct(self, pred_labels, labels, weights=None):
        # weights is np array, so convert it to tensor and ensure the device
        loss_fct = nn.MSELoss(reduction=self.args.loss_reduction)
        mask_value = -1
        mask = torch.not_equal(labels, mask_value)
        if weights is not None:
            if self.args.loss_reduction == 'mean':
                loss = (weights.expand_as(pred_labels) * (pred_labels - labels) ** 2).mean()
            else:
                loss = (weights.expand_as(pred_labels) * (pred_labels - labels) ** 2).sum()
        else:
            loss = loss_fct(pred_labels * mask, labels * mask)
        return loss
    
    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        nn.init.xavier_uniform_(linear.weight)
        # nn.init.normal_(linear.weight, std=std)
        if bias:
            nn.init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        act_func = nn.ReLU
        if self.args.act_func == 'tanh':
            act_func = nn.Tanh
        elif self.args.act_func == 'sigmoid':
            act_func = nn.Sigmoid
        elif self.args.act_func == 'elu':
            act_func = nn.ELU
        ffnn = [self.make_linear(feat_size, hidden_size[0]), act_func(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), act_func(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

