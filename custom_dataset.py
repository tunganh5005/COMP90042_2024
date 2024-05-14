
import json

import torch
from torch.utils.data import Dataset, DataLoader

from preprocess import preprocess_sentence

from word_embedding import embed




class DocumentDataset(Dataset):
    def __init__(self, file_name, embedding_size, document_target_length, query_target_length):
        self.data = []
        self.document_target_length = document_target_length
        self.query_target_length = query_target_length
        self.embedding_size = embedding_size

        # Opening JSON file 
        with open(file_name, 'rb') as f:
            # Load JSON content
            self.data = json.load(f)
        
        
    def __len__(self):
        return len(self.data)
    
    def pad_sequence_start(self, sequence, target_length):
        """
        Adds zero padding to the beginning of a sequence to achieve the target length.

        Args:
        sequence (list): The original sequence to be padded.
        target_length (int): The desired length of the sequence after padding.
        pad_value (int, optional): The value used for padding. Default is 0.

        Returns:
        list: A new list padded at the beginning to the specified target length.
        """
        current_length = len(sequence)
        if current_length >= target_length:
            return sequence
        # Calculate how many zeros to add to the beginning
        num_padding = target_length - current_length
        # Create a new list with zeros at the beginning and the original sequence
        padded_sequence = [torch.zeros(self.embedding_size)] * num_padding + sequence
        return padded_sequence
    

    def __getitem__(self, idx):
        

        item = self.data[idx]

        query = preprocess_sentence(item['query'])

        query_embeddings = []

        for token in query:
            try:
                query_embeddings.append(torch.tensor(embed(token)))
            except:
                query_embeddings.append(torch.zeros(self.embedding_size))

        query_embeddings = torch.stack(self.pad_sequence_start(query_embeddings, self.query_target_length))

        document = preprocess_sentence(item['document'])

        document_embeddings = []

        for token in document:
            try:
                document_embeddings.append(torch.tensor(embed(token)))
            except:
                document_embeddings.append(torch.zeros(self.embedding_size))

        document_embeddings = torch.stack(self.pad_sequence_start(document_embeddings, self.document_target_length))

        label = item['label']

        label_to_index = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT_ENOUGH_INFO': 2, 'DISPUTED': 3}

        # Encode labels
        label = torch.tensor(label_to_index[item['label']], dtype=torch.long)

        return query_embeddings, document_embeddings, label
    

if __name__ == '__main__':
    dataset = DocumentDataset('data/train_dataset.json', embedding_size=100, target_length=50)
    print(dataset[3][0].shape)
