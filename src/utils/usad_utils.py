import numpy as np
import pandas as pd

import csv

from src.utils.utils import *


def evaluate(model, device, val_loader, n):
    outputs = [model.validation_step(to_device(batch,device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)

    
def training(epochs, model, device, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch=to_device(batch,device)
            
            #Train AE1            
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            #Train AE2
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
        result = evaluate(model, device, val_loader, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def testing(model, device, test_loader, alpha=.5, beta=.5):
    results=[]
    for [batch] in test_loader:
        batch=to_device(batch,device)
        w1=model.decoder1(model.encoder(batch))
        w2=model.decoder2(model.encoder(w1))
        results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
    return results


def test_model(model, device, test_loader, alpha=0.5, beta=0.5):
    with torch.no_grad(): 
        results = testing(model, device, test_loader, alpha=0.5, beta=0.5)
    
    return results

def get_prediction_score(results):
    score = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                  results[-1].flatten().detach().cpu().numpy()])
    return score


def save_model(model, path="models/model.pth"):
    torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
        }, path)
    

def load_checkpoint(model, path="models/model.pth"):
    checkpoint = torch.load(path)

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder1.load_state_dict(checkpoint['decoder1'])
    model.decoder2.load_state_dict(checkpoint['decoder2'])
    
    return model


def save_labels(labels, path='/notebooks/results/labels/'):
    filename = 'labels.csv'
    with open(path + filename, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(labels)