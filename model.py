import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        # Define the properties
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # The linear layer maps the output of the LSTM into the size of the vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.word_embeddings(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_outputs, _ = self.lstm(inputs)
        outputs = self.fc(lstm_outputs)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_sentence = []
        for i in range(max_len):
            lstm_outputs, states = self.lstm(inputs, states)
            outputs = self.fc(lstm_outputs.squeeze(1))
            _, predicted = outputs.max(1)
            predicted_sentence.append(predicted.item())
            inputs = self.word_embeddings(predicted).unsqueeze(1)
        return predicted_sentence