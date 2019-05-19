import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import random
import spacy
import pickle

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_en = spacy.load('en')

def tokenizer(text):
    token = [t.text for t in spacy_en.tokenizer(text)]
    if len(token) < 3:
        for i in range(0, 5 - len(token)):
            token.append('<PAD>')
    return token


TEXT = data.Field(tokenize=tokenizer)
LABEL = data.LabelField()

train_data, valid_data, test_data = data.TabularDataset.splits(
    path='data', train='train.csv',
    validation='valid.csv', test='test.csv',
    format='csv', skip_header=True,
    csv_reader_params={'delimiter':'\t'},
    fields=[('text',TEXT),('label',LABEL)]
)

print('train_data[0]', vars(train_data[0]))

TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

print(LABEL.vocab.stoi)

BATCH_SIZE = 64

device = torch.device('cuda')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),
    device=device)

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
    
    def forward(self, encoder_outputs):
        # encoder_outputs = [batch size, sent len, hid dim]
        energy = self.projection(encoder_outputs)
        # energy = [batch size, sent len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # weights = [batch size, sent len]
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        # outputs = [batch size, hid dim]
        return outputs, weights


# use only 3-gram filter and bidirection
class C_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_size, bidirectional, hidden_dim , output_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size,embedding_dim)) 
        self.lstm = nn.LSTM(n_filters, hidden_dim, bidirectional=bidirectional, dropout=dropout)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x): 
        #x = [sent len, batch size]
        x = x.permute(1, 0)   
        #x = [batch size, sent len]
        embedded = self.embedding(x)     
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = F.relu(self.conv(embedded)).squeeze(3)
        #conved = [batch size, n_filters, sent len - filter_size]
        x = conved.permute(2, 0, 1)
        #x = [sent len- filter_size, batch size, n_filters]
        output, (hidden, cell) = self.lstm(x)
        #output = [sent len - filter_size, batch size, hid_dim*num directional]
        output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        # output = [sent len, batch size, hid dim]
        ouput = output.permute(1, 0, 2)
        # ouput = [batch size, sent len, hid dim]
        new_embed, weights = self.attention(ouput)
        # new_embed = [batch size, hid dim]
        # weights = [batch size, sent len]
        new_embed = self.dropout(new_embed)
        return self.fc(new_embed)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 150
N_FILTERS = 150
FILTER_SIZE = 3
BIDIRECTIONAL = True
OUTPUT_DIM = len(LABEL.vocab)
DROPOUT = 0.5

model = C_LSTM(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZE, BIDIRECTIONAL, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)


pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum()/torch.FloatTensor([y.shape[0]])

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text)
        
        loss = criterion(predictions, batch.label)
        
        acc = categorical_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text)
            
            loss = criterion(predictions, batch.label)
            
            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

N_EPOCHS = 5

for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

torch.save(model, "c_lstm_attention")

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')