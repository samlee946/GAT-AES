import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import process_data
from torch_geometric.nn import GCNConv, GATConv

from torch_geometric.nn import GCNConv

class GNN(nn.Module):
    def __init__(self, args, encoded_prompts):
        super().__init__()
        self.args = args
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
        self.emb_transform = nn.Linear(self.emb_size, self.args.GNN_hidden_size) if self.num_emb_nodes > 0 else None
        self.feat_transform = nn.Linear(self.num_features, self.args.GNN_hidden_size) if self.num_feat_nodes > 0 else None
        
        # Trait nodes (initialized in hidden_size dimension)
        self.trait_nodes = nn.Parameter(torch.randn(self.num_traits, self.args.GNN_hidden_size))
        
        # GAT layers
        self.gat_layers = nn.ModuleList()

        # First GAT layer
        self.gat_layers.append(GATConv(
            in_channels=self.args.GNN_hidden_size,
            out_channels=self.args.GNN_hidden_size // self.args.GNN_num_heads,
            heads=self.args.GNN_num_heads,
            concat=True,
            dropout=self.args.GNN_dropout
        ))
        
        # Middle GAT layers
        for _ in range(self.args.GNN_num_layers - 2):
            self.gat_layers.append(GATConv(
                in_channels=self.args.GNN_hidden_size,
                out_channels=self.args.GNN_hidden_size // self.args.GNN_num_heads,
                heads=self.args.GNN_num_heads,
                concat=True,
                dropout=self.args.GNN_dropout
            ))
        
        # Last GAT layer
        if self.args.GNN_num_layers > 1:
            self.gat_layers.append(GATConv(
                in_channels=self.args.GNN_hidden_size,
                out_channels=self.args.GNN_hidden_size,
                heads=1,
                concat=False,
                dropout=self.args.GNN_dropout
            ))
        
        # # Output layer
        # self.output_layer = nn.Linear(self.args.GNN_hidden_size, 1)

        # Separate output layers for each trait
        self.output_layers = nn.ModuleList([nn.Linear(self.args.GNN_hidden_size, 1) for _ in range(self.num_traits)])
        
        # Create static edge structure
        self.register_buffer('edge_index', self._create_edge_structure())

    def _create_edge_structure(self):
        edges = []

        # Calculate start indices for each type of node
        trait_start = 0
        emb_start = self.num_traits
        feat_start = emb_start + self.num_emb_nodes
        
        if self.num_emb_nodes > 0:
            # Embedding-trait edges
            for i in range(self.num_traits):
                edges.append([emb_start, i])
            
        if self.num_features > 0 and self.num_feat_nodes > 0:
            # Feature-trait edges
            for i in range(self.num_traits):
                edges.append([feat_start, i])
            
        if self.args.GNN_edge_mode == 'all_pairs':
            # Trait-trait edges (fully connected between traits)
            for i in range(self.num_traits):
                for j in range(self.num_traits):
                    if i != j:
                        edges.append([i, j])
        elif self.args.GNN_edge_mode == 'none':
            pass
        elif self.args.GNN_edge_mode == 'arts_order':
            arts_order = ['voice', 'style', 'sentence_fluency', 'word_choice', 'conventions', 'organization', 'narrativity', 'language', 'prompt_adherence', 'content', 'overall_resolved']
            arts_order_to_node_idx = [self.score_vector_positions[trait] for trait in arts_order]
            # connect each trait with every previous trait
            for i in range(1, len(arts_order_to_node_idx)):
                for j in range(i):
                    edges.append([arts_order_to_node_idx[i], arts_order_to_node_idx[j]])
        
        return torch.tensor(edges, dtype=torch.long).t()

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

        if self.num_emb_nodes > 0:
            # get essay embeddings
            essay_input_ids = essay_input_ids.squeeze(1) # shape (batch_size, seq_len)
            essay_attention_mask = essay_attention_mask.squeeze(1) # shape (batch_size, seq_len)
            if essay_token_type_ids is not None:
                essay_token_type_ids = essay_token_type_ids.squeeze(1) # shape (batch_size, seq_len)
                lm_outputs_essay = self.lm(input_ids=essay_input_ids, attention_mask=essay_attention_mask, token_type_ids=essay_token_type_ids, output_hidden_states=True)
            else:
                lm_outputs_essay = self.lm(input_ids=essay_input_ids, attention_mask=essay_attention_mask, output_hidden_states=True)
            ## essay embeddings by [CLS]
            essay_embeddings = lm_outputs_essay.last_hidden_state[:, 0, :] # shape (batch_size, hidden_size)
            
            # Transform inputs
            emb_transformed = self.emb_transform(essay_embeddings)  # [batch_size, GNN_hidden_size]
        if self.num_features > 0 and self.num_feat_nodes > 0:
            feat_transformed = self.feat_transform(essay_feats)  # [batch_size, GNN_hidden_size]
        
        preds = []

        # import ipdb; ipdb.set_trace()

        for i in range(batch_size):
            # Construct node features for this instance
            instance = [self.trait_nodes, ]
            if self.num_emb_nodes > 0:
                instance.append(emb_transformed[i:i+1])
            if self.num_features > 0 and self.num_feat_nodes > 0:
                instance.append(feat_transformed[i:i+1])
            x = torch.cat(instance, dim=0)
            
            # Apply GAT layers
            for gat_layer in self.gat_layers:
                x = F.relu(gat_layer(x, self.edge_index))
            
            # Get trait node representations
            trait_outputs = x[:self.num_traits]  # [num_traits, GNN_hidden_size]
            
            # Generate predictions for each trait
            # logits = self.output_layer(trait_outputs).squeeze(-1)  # [num_traits]

            # Use separate output layers for each trait
            logits = torch.stack([
                self.output_layers[j](trait_outputs[j]) 
                for j in range(self.num_traits)
            ]).squeeze(-1)

            preds.append(logits)
        
        # import ipdb; ipdb.set_trace()

        preds = torch.stack(preds)  # [batch_size, num_traits]
        preds = torch.sigmoid(preds)

        if labels is None:
            return None, preds

        loss = self.mask_loss_fct(preds, labels, self.trait_weights_for_loss)

        return loss, preds

        # hidden_states = torch.stack(lm_outputs_essay.hidden_states, dim=0)  # Shape: [num_layers, batch_size, seq_len, args.GNN_hidden_size]
        # hidden_states = hidden_states[:, :, 0]  # Shape: [num_layers, batch_size, args.GNN_hidden_size]
        # # for each trait, get the embeddings
        # for trait in self.score_vector_positions.keys():
        #     if self.args.layer_weight_scheme == 'simple':
        #         layer_weights = getattr(self, f'layer_weights_{trait}')
        #         layer_weights = F.softmax(layer_weights, dim=0)
        #         trait_embedding = torch.einsum("l,lbd->bd", layer_weights, hidden_states)  # Shape: [batch_size, seq_len, args.GNN_hidden_size]
        #     elif self.args.layer_weight_scheme == 'prompt_attention':
        #         trait_query_embeddings = getattr(self, f'trait_query_embeddings_{trait}') # [1, hidden_size]
        #         trait_query_embeddings = trait_query_embeddings.expand(essay_input_ids.size(0), -1) # [batch_size, hidden_size]
        #         query = trait_query_embeddings + prompt_embeddings
        #         d_k = hidden_states.size(-1) ** 0.5
        #         attention_scores = torch.einsum("bd,lbd->bl", query, hidden_states) / d_k  # [batch_size, num_layers]
        #         attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)  # Shape: [batch_size, num_layers]
        #         trait_embedding = torch.einsum("bl,lbd->bd", attention_weights, hidden_states)  # Shape: [batch_size, args.GNN_hidden_size]
        #     else:
        #         raise NotImplementedError

        #     if self.args.dim_reduction > 0:
        #         trait_embedding = getattr(self, f'dim_reduction_{trait}')(trait_embedding)

        #     if self.args.apply_max_pooling:
        #         trait_embedding = trait_embedding.unsqueeze(1)
        #         trait_embedding = self.pooling_layer(trait_embedding).squeeze(1)

        #     if self.args.use_essay_feats:
        #         trait_embedding = torch.cat([trait_embedding, essay_feats], dim=1)

        #     ffnn = getattr(self, f'ffnn_{trait}')
        #     logits = ffnn(trait_embedding)
        #     pred_score = self.sigmoid(logits).squeeze(1)
        #     preds.append(pred_score)

        # preds = torch.stack(preds, dim=1)

        # if labels is None:
        #     return None, preds

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


# create a subclass of GNN for multi-node GNN
class MultiNodeGNN(GNN):
    def __init__(self, args, encoded_prompts):
        
        self.num_emb_nodes = args.GNN_num_emb_nodes
        self.num_feat_nodes = args.GNN_num_feat_nodes

        super().__init__(args, encoded_prompts)
        
        # Initial transformations - now handle multiple nodes
        # For embeddings: transform from emb_size to hidden_size for each embedding node
        self.emb_transform = nn.ModuleList([
            nn.Linear(self.emb_size, args.GNN_hidden_size) for _ in range(args.GNN_num_emb_nodes)
        ]) if self.num_emb_nodes > 0 else None
        
        if self.num_features > 0 and self.num_feat_nodes > 0:
            # For features: transform from feat_size to hidden_size for each feature node
            self.feat_transform = nn.ModuleList([
                nn.Linear(self.num_features, args.GNN_hidden_size) for _ in range(args.GNN_num_feat_nodes)
            ]) 


    def _create_edge_structure(self):
        edges = []
        
        # Calculate start indices for each type of node
        trait_start = 0
        emb_start = self.num_traits
        feat_start = emb_start + self.num_emb_nodes
        
        # Embedding-trait edges
        for i in range(self.num_emb_nodes):
            emb_idx = emb_start + i
            for j in range(self.num_traits):
                edges.append([emb_idx, j])
            
        if self.num_features > 0:
            # Feature-trait edges
            for i in range(self.num_feat_nodes):
                feat_idx = feat_start + i
                for j in range(self.num_traits):
                    edges.append([feat_idx, j])
            
        if self.args.GNN_edge_mode == 'all_pairs':
            # Trait-trait edges (fully connected between traits)
            for i in range(self.num_traits):
                for j in range(self.num_traits):
                    if i != j:
                        edges.append([i, j])
        elif self.args.GNN_edge_mode == 'none':
            pass
        elif self.args.GNN_edge_mode == 'arts_order':
            arts_order = ['voice', 'style', 'sentence_fluency', 'word_choice', 'conventions', 'organization', 'narrativity', 'language', 'prompt_adherence', 'content', 'overall_resolved']
            arts_order_to_node_idx = [self.score_vector_positions[trait] for trait in arts_order]
            # connect each trait with every previous trait
            for i in range(1, len(arts_order_to_node_idx)):
                for j in range(i):
                    edges.append([arts_order_to_node_idx[i], arts_order_to_node_idx[j]])
        
        return torch.tensor(edges, dtype=torch.long).t()
        
    def forward(self, essay_ids, prompt_ids, essay_input_ids, essay_attention_mask, essay_token_type_ids, essay_feats, labels=None):

        batch_size = essay_ids.shape[0]

        if self.num_emb_nodes > 0:
            # get essay embeddings
            essay_input_ids = essay_input_ids.squeeze(1) # shape (batch_size, seq_len)
            essay_attention_mask = essay_attention_mask.squeeze(1) # shape (batch_size, seq_len)
            if essay_token_type_ids is not None:
                essay_token_type_ids = essay_token_type_ids.squeeze(1) # shape (batch_size, seq_len)
                lm_outputs_essay = self.lm(input_ids=essay_input_ids, attention_mask=essay_attention_mask, token_type_ids=essay_token_type_ids, output_hidden_states=True)
            else:
                lm_outputs_essay = self.lm(input_ids=essay_input_ids, attention_mask=essay_attention_mask, output_hidden_states=True)
            ## essay embeddings by [CLS]
            essay_embeddings = lm_outputs_essay.last_hidden_state[:, 0, :] # shape (batch_size, hidden_size)

            # Transform inputs
            emb_transformed = []
            for i in range(self.num_emb_nodes):
                emb_transformed.append(self.emb_transform[i](essay_embeddings))

        if self.num_features > 0:
            feat_transformed = []
            for i in range(self.num_feat_nodes):
                feat_transformed.append(self.feat_transform[i](essay_feats))
        
        preds = []

        # import ipdb; ipdb.set_trace()

        for i in range(batch_size):
            # Construct node features for this instance
            instance = [self.trait_nodes, ]

            for _node_idx in range(self.num_emb_nodes):
                instance.append(emb_transformed[_node_idx][i:i+1])

            if self.num_features > 0:
                for _node_idx in range(self.num_feat_nodes):
                    instance.append(feat_transformed[_node_idx][i:i+1])
            
            x = torch.cat(instance, dim=0)

            # Apply GAT layers
            for gat_layer in self.gat_layers:
                x = F.relu(gat_layer(x, self.edge_index))
            
            # Get trait node representations
            trait_outputs = x[:self.num_traits]  # [num_traits, GNN_hidden_size]
            
            # Generate predictions for each trait
            # logits = self.output_layer(trait_outputs).squeeze(-1)  # [num_traits]

            # Use separate output layers for each trait
            logits = torch.stack([
                self.output_layers[j](trait_outputs[j]) 
                for j in range(self.num_traits)
            ]).squeeze(-1)

            preds.append(logits)
        
        # import ipdb; ipdb.set_trace()

        preds = torch.stack(preds)  # [batch_size, num_traits]
        preds = torch.sigmoid(preds)

        if labels is None:
            return None, preds

        loss = self.mask_loss_fct(preds, labels, self.trait_weights_for_loss)

        return loss, preds

