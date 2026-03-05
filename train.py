import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim


#1 Load JSON dataset
# ===============================
with open("happy_hour_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)



#2 TF-IDF Vectorization
# ===============================
all_texts = [t for texts in X_sets for t in texts]
all_urls = [u for urls in sub_url_sets for u in urls]

text_vectorizer = TfidfVectorizer(max_features=4000)
text_vectorizer.fit(all_texts)

url_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,5), max_features=1000)
url_vectorizer.fit(all_urls)

X_vectorized = []
for texts, urls in zip(X_sets, sub_url_sets):
    text_features = text_vectorizer.transform(texts).toarray()
    url_features = url_vectorizer.transform(urls).toarray()
    combined = np.hstack([text_features, url_features])
    X_vectorized.append(combined)

# Feature normalization
scaler = StandardScaler()
X_vectorized_scaled = [scaler.fit_transform(x) for x in X_vectorized]


#3 Dataset Class
# ===============================
class BaseURLDataset(Dataset):
    def __init__(self, X_sets, y_sets, sub_url_sets):
        self.X_sets = X_sets
        self.y_sets = y_sets
        self.sub_url_sets = sub_url_sets

    def __len__(self):
        return len(self.X_sets)

    def __getitem__(self, idx):
        X = torch.tensor(self.X_sets[idx], dtype=torch.float32)
        y = torch.tensor(self.y_sets[idx], dtype=torch.long)
        urls = self.sub_url_sets[idx]
        return X, y, urls


#4 Split train/test
# ===============================
train_idx, test_idx = train_test_split(range(len(X_vectorized_scaled)), test_size=0.2, random_state=42)

train_dataset = BaseURLDataset([X_vectorized_scaled[i] for i in train_idx],
                               [y_sets[i] for i in train_idx],
                               [sub_url_sets[i] for i in train_idx])

test_dataset = BaseURLDataset([X_vectorized_scaled[i] for i in test_idx],
                              [y_sets[i] for i in test_idx],
                              [sub_url_sets[i] for i in test_idx])

trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


#5 FFN Model
# ===============================
class SetFFN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, X):
        x = torch.relu(self.fc1(X))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x).squeeze(-1)

input_dim = X_vectorized_scaled[0].shape[1]
net = SetFFN(input_dim)


#6 Loss & Optimizer (SGD with momentum)
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#7 Training Loop
# ===============================
epochs = 150
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for X_set, y_set, _ in trainloader:
        X_set = X_set.squeeze(0)  
        y_set = y_set.squeeze(0) 

        optimizer.zero_grad()
        outputs = net(X_set)    

        # CrossEntropyLoss 
        loss = criterion(outputs.unsqueeze(0), torch.argmax(y_set).unsqueeze(0))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Training accuracy
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=0)  
            pred_idx = torch.argmax(probs).item()
            true_idx = torch.argmax(y_set).item()
            correct_train += int(pred_idx == true_idx)
            total_train += 1

    print(f"Epoch {epoch+1}/{epochs} -> Loss: {running_loss/len(trainloader):.4f} | "
          f"Train Accuracy: {100*correct_train/total_train:.2f}%")



#8 Save Model
# ===============================
PATH = './happyhour_ffn_with_url.pth'
torch.save(net.state_dict(), PATH)
print("Model saved!")


#9 Evaluation (Test set)
# ===============================
net.eval()
correct = 0
total = 0
print("\n===== PER BASE URL RESULTS =====\n")

with torch.no_grad():
    for X_set, y_set, sub_urls in testloader:
        X_set = X_set.squeeze(0)
        y_set = y_set.squeeze(0)

        outputs = net(X_set)
        probs = torch.softmax(outputs, dim=0)
        pred_idx = torch.argmax(probs).item()
        pred_sub_url = sub_urls[pred_idx]
        true_idx = torch.argmax(y_set).item()
        true_sub_url = sub_urls[true_idx]
        is_correct = int(pred_sub_url == true_sub_url)

        print(f"BASE URL:")
        print(f"  Predicted: {pred_sub_url}  (prob={probs[pred_idx]:.3f})")
        print(f"  Actual:    {true_sub_url}")
        print(f"  {'✔ CORRECT' if is_correct else '✘ WRONG'}\n")

        correct += is_correct
        total += 1

print(f"Per-base-URL Sub-Page Accuracy: {100 * correct/total:.2f}%")
