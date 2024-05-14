
import torch
import torch.nn as nn

from torch import optim

from models import QueryDocumentGRU


from custom_dataset import DocumentDataset

from torch.utils.data import DataLoader

# Create the model instance
model = QueryDocumentGRU(embedding_dim=100, gru_hidden_size=512, num_classes=4)

dataset = DocumentDataset('data/train_dataset.json', embedding_size=100, query_target_length=100, document_target_length=400)



train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  # Applies Softmax internally for stability and efficiency
optimizer = optim.Adam(model.parameters(), lr=0.001)




def train(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        total_loss = 0
        print('Epoch - ', epoch+1)
        model.train()
        losses = []
        for queries, documents, labels in train_loader:

            outputs = model(queries, documents)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear existing gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters

            losses.append(loss.item())

            print(f"Batch {len(losses)}/{len(train_loader)}")

        # Print average loss for the epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {sum(losses) / len(losses)}')

        # Save model weights
        torch.save(model.state_dict(), 'weights/gru.pth')


if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer, num_epochs=10)

        