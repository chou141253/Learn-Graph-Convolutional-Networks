import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt

from gcn import SimpleGCN

DATA_PATH = "datas.txt"
LABEL_PATH = "labels.txt"
MAX_EPOCHS = 30000
COLORS = ['green', 'red', 'blue', 'purple']

def extract_datas(data_path: str, label_path: str):

    # load data
    with open(data_path, "r") as ftxt:
        datas = ftxt.read().split('\n')

    # load label
    with open(label_path, "r") as ftxt:
        labels = ftxt.read().split('\n')
    labels = [int(x) for x in labels]
    
    print("total nodes: ", len(datas))
    print("total labels:", len(labels))

    adjacency_matrix = []
    for data in datas:
        conns = data.split(' ')
        adjacency_matrix.append([int(x) for x in conns])

    adjacency_matrix = torch.tensor(adjacency_matrix)
    labels = torch.tensor(labels)
    print("Adjacency Matrix:", adjacency_matrix.size())

    return adjacency_matrix, labels


def visualize(adjacency_matrix, labels):

    labels_dict = {}
    colors = []
    for i in range(labels.size(0)):
        labels_dict[str(i)] = i
        colors.append(COLORS[int(labels[i])])

    rows, cols = torch.where(adjacency_matrix == 1)
    rows = [str(x) for x in rows.tolist()]
    cols = [str(x) for x in cols.tolist()]
    edges = zip(rows, cols)

    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, node_color=colors, labels=labels_dict, with_labels=True)
    plt.show()
        


def train(epoch, model, optimizer, criterion, data, label):
    model.train()
    out = model(data)
    # out: (1, 4, 34)
    # ---> (34, 4)
    out = out.squeeze(0).transpose(1, 0)
    loss = criterion(out, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # msic...
    if (epoch+1) % 1000 == 0:
        corrects = torch.max(out, dim=-1)[1].eq(label).cpu().sum().item()
        train_acc = float(corrects / out.size(0))

        print("epoch:{:.1E}, loss:{}, accuracy:{}".format(
            epoch+1, float(loss), round(100*train_acc, 2)
        ))



if __name__ == "__main__":
    adjacency_matrix, labels = extract_datas(DATA_PATH, LABEL_PATH)
    visualize(adjacency_matrix, labels)

    # ======== build data, label and model... ========
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_nodes = adjacency_matrix.size(0)
    adjacency_matrix = adjacency_matrix + torch.eye(num_nodes)
    adjacency_matrix = adjacency_matrix / 34
    # unsqueeze(0): add batch dimension (for pytorch training).
    # unsqueeze(0): add channels.
    datas = torch.ones(num_nodes).unsqueeze(0).unsqueeze(0)

    datas, labels = datas.to(device), labels.to(device)

    model = SimpleGCN(
        adj=adjacency_matrix.float().to(device),
        num_class=4)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(MAX_EPOCHS):
        train(epoch, model, optimizer, criterion, datas, labels)
