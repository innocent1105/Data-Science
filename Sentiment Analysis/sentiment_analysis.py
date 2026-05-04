import torch
import torch.nn as nn
import numpy as np

# small dataset, i have one in csv
sentences = [
    "i love this", "great movie", "absolutely wonderful", "best day ever",
    "i hate this", "terrible movie", "absolutely awful", "worst day ever",
    "this is good", "not bad at all", "pretty enjoyable", "quite nice",
    "this is bad", "not good at all", "pretty boring", "quite awful",
]
labels = [1,1,1,1, 0,0,0,0, 1,1,1,1, 0,0,0,0]  # 1=positive, 0=negative
# i attached each sentence to a value, 1 or 0

#  assign every unique word an integer id
vocab = {}
for sentence in sentences:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = len(vocab)

print(f"Vocabulary size: {len(vocab)} words")
print(dict(list(vocab.items())[:6]), "...")

# represent each sentence as a vector
# each position = a word in vocab, value = how many times it appears
def sentence_to_vector(sentence, vocab):
    vec = np.zeros(len(vocab))
    for word in sentence.split():
        if word in vocab:
            vec[vocab[word]] += 1
    return vec

X = torch.tensor([sentence_to_vector(s, vocab) for s in sentences], dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

print(f"\nInput shape: {X.shape}")  # (16 sentences, vocab_size)

# model configuration
model = nn.Sequential(
    nn.Linear(len(vocab), 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

# here we train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

for epoch in range(500):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# Testing
def predict(sentence):
    vec = torch.tensor(sentence_to_vector(sentence, vocab), dtype=torch.float32)
    score = model(vec).item()
    label = "positive " if score > 0.5 else "negative "
    print(f"  '{sentence}' → {label} ({score:.2f})")

print("\n--- Try some sentences ---")
predict("i love this movie")
predict("i hate this day")
predict("wonderful and great")
predict("awful and terrible")
predict("this is good not bad")  # tricky — mixed words
