import torch
import torch.nn as nn

import torch.nn.functional as F

class QueryDocumentGRU(nn.Module):
    def __init__(self, embedding_dim, gru_hidden_size, num_classes):
        super(QueryDocumentGRU, self).__init__()
        
        
        # GRU layers for the query and the document
        self.query_gru = nn.GRU(embedding_dim, gru_hidden_size, batch_first=True)
        self.document_gru = nn.GRU(embedding_dim, gru_hidden_size, batch_first=True)
        
        # Dense layer for classification
        self.fc = nn.Linear(2 * gru_hidden_size, num_classes)  # Assuming concatenation of GRU outputs

    def forward(self, query, document, inference=False):
        
        # Passing through GRU layers
        _, query_hidden = self.query_gru(query)  # We use only the last hidden state
        _, document_hidden = self.document_gru(document)  # We use only the last hidden state
        
        # Concatenating the final hidden states of both GRU layers
        combined = torch.cat((query_hidden[-1], document_hidden[-1]), dim=1)
        
        # Classification layer
        logits = self.fc(combined)

        if inference:
            return F.softmax(logits, dim=1)  # Apply softmax for inference
        else:
            return logits  # Return raw logits for training
    

