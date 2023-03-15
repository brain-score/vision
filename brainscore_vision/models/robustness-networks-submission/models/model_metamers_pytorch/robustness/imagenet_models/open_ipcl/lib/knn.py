import time
import torch
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
# import nethook as nethook
from lib import nethook
from IPython.core.debugger import set_trace
import torch.nn.functional as F 

try:
    from fastprogress.fastprogress import progress_bar
except:
    from fastprogress import progress_bar
    
def get_features(model, dataloader, layer_name, device=None, out_device=None):    
    if not isinstance(model, nethook.InstrumentedModel):
        model = nethook.InstrumentedModel(model)
    model.retain_layers([layer_name])
    features,labels,indexes = [],[],[]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if out_device is None:
        out_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    model.to(device)
    model.eval()
    with torch.no_grad():
        for imgs,targs,idxs in progress_bar(dataloader):
            out = model(imgs.to(device))
            X = model.retained_layer(layer_name)
            X = F.normalize(X, dim=1)
            X = X.view(X.shape[0], -1)
            features.append(X.to(out_device))
            labels.append(targs.to(out_device))
            indexes.append(idxs.to(out_device))
    
    features = torch.cat(features)
    labels = torch.cat(labels)
    indexes = torch.cat(indexes)
    
    return features, labels, indexes

def do_kNN(trainFeatures, trainLabels, testFeatures, testLabels, C, K, sigma, device=None, out_device=None):
    '''
        trainFeatures: [nTrainSamples, nFeatures]
        trainLabels: [nTrainSamples]
        
        testFeatures: [nTestSamples, nFeatures]
        testLabels: [nTestSamples]
    '''
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'            
    
    dist = torch.mm(testFeatures, trainFeatures.T).to(device)
    
    batchSize = len(testLabels)
    
    yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
    
    candidates = trainLabels.view(1,-1).expand(batchSize, -1)
    
    retrieval = torch.gather(candidates, 1, yi)
    retrieval_one_hot = torch.zeros(K, C).to('cpu')
    retrieval_one_hot.resize_(batchSize * K, C).zero_()
    retrieval_one_hot.scatter_(1, retrieval.view(-1, 1).cpu(), 1)
    yd_transform = yd.clone().div_(sigma).exp_()
    probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1).cpu()), 1)
    _, predictions = probs.sort(1, True)
    
    # Find which predictions match the target
    correct = predictions.eq(testLabels.view(-1,1).cpu())
    correct.shape
    
    total = correct.size(0)
    top1 = correct.narrow(1,0,1).sum().item() / total * 100
    top5 = correct.narrow(1,0,5).sum().item() / total * 100
    
    return top1, top5, total

def repeated_kfold_kNN(model, dataloader, layer_name, K=40, sigma=.07, n_splits=5, n_repeats=5, 
                       random_state=123456, epoch=None):
    
    print("\n================================")
    print(' Generalization Epoch: %d' % epoch)
    print("================================")
    
    print(f"running repeated k-fold validation (n_splits={n_splits}, n_repeats={n_repeats})")
    print(f"extracting features (layer_name='{layer_name}')...")
    features, labels, indexes = get_features(model, dataloader, layer_name)
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    C = labels.max() + 1
    num_splits = n_splits * n_repeats
    print(f"performing cross validated kNN accuracy (K={K}, sigma={sigma})...")
    TOP1, TOP5 = [], []
    for split_num, (train_index, test_index) in enumerate(progress_bar(rkf.split(features.cpu(),labels.cpu()), total=num_splits)):
        trainFeatures = features[train_index]
        trainLabels = labels[train_index]

        testFeatures = features[test_index]
        testLabels = labels[test_index]
        
        top1, top5, _ = do_kNN(trainFeatures, trainLabels, testFeatures, testLabels, C, K, sigma)
        print(f"[{split_num}/{num_splits}]: top1={top1:3.2f}, top5={top5:3.2f}")
        
        TOP1.append(top1)
        TOP5.append(top5)
    TOP1 = torch.tensor(TOP1)
    TOP5 = torch.tensor(TOP5)
    
    print(f"Average Top1 % = {TOP1.mean():4.2f}%, Top5 = {TOP5.mean():4.2f}%")
    return TOP1, TOP5

def run_kNN(model, train_loader, test_loader, layer_name, K=200, sigma=.07, num_chunks=200, out_device=None):
    print("extracting training features...")
    trainFeatures, trainLabels, indexes = get_features(model, train_loader, layer_name, out_device=out_device)
    print("extracting test features...")
    testFeatures, testLabels, indexes = get_features(model, test_loader, layer_name, out_device=out_device)
    print("running kNN test...")
    
    # split test features into chunks to avoid out-of-memory error:
    chunkFeatures = torch.chunk(testFeatures, num_chunks, dim=0)
    chunkLabels = torch.chunk(testLabels, num_chunks, dim=0)

    C = trainLabels.max() + 1
    top1, top5, total = 0., 0., 0.
    for features, labels in progress_bar(zip(chunkFeatures, chunkLabels), total=num_chunks):
        top1_, top5_, total_ = do_kNN(trainFeatures, trainLabels, features, labels, C, K, sigma)
        top1 += top1_ / 100 * total_
        top5 += top5_ / 100 * total_
        total += total_
    top1 = top1 / total * 100
    top5 = top5 / total * 100
    
    print(f"run_kNN accuracy: top1={top1}, top5={top5}")
    
    return top1, top5

def gen_features(model, dataloader, layer_name, device=None, out_device=None):    
    if not isinstance(model, nethook.InstrumentedModel):
        model = nethook.InstrumentedModel(model)
    model.retain_layers([layer_name])
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if out_device is None:
        out_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    model.to(device)
    model.eval()
    with torch.no_grad():
        for imgs,targs,idxs in progress_bar(dataloader):
            out = model(imgs.to(device))
            X = model.retained_layer(layer_name)
            X = F.normalize(X, dim=1)
            X = X.view(X.shape[0], -1)
            yield X.to(out_device), targs.to(out_device), idxs.to(out_device)
    

def run_kNN_chunky(model, train_loader, test_loader, layer_name, K=200, sigma=.07, num_chunks=10, out_device=None):
    '''this version scales better to larger feature spaces
    
        we compute the full testFeatures, testLabels,
        
        then we iterate over the training set in batches, accumulating `num_chunks` (should
        be `num_batches`, but keeping the naming the same as run_kNN for api consistency).
        
    '''
    if out_device is None:
        out_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    print("extracting test features...")
    testFeatures, testLabels, indexes = get_features(model, test_loader, layer_name)
    C = testLabels.max() + 1
    
    print("extracting/comparing to train features...")
    dist = None
    trainLabels = None
    trainIndexes = None
    
    #for batch_num, (trainFeatures, labels, indexes) in enumerate(gen_features(model, train_loader, layer_name)):
    trainFeatures, labels, indexes = None, None, None
    for batch_num, (feat, lab, ind) in enumerate(gen_features(model, train_loader, layer_name)): 
        
        trainFeatures = feat if trainFeatures is None else torch.cat([trainFeatures, feat])
        labels = lab if labels is None else torch.cat([labels, lab])
        indexes = ind if indexes is None else torch.cat([indexes, ind])
        
        # accumulate a few batches before proceeding
        if (batch_num+1) % num_chunks != 0 and batch_num != (len(train_loader)-1):
            continue
        
        # compute distances, concat with retained
        d = torch.mm(testFeatures, trainFeatures.T).to(out_device)        
        dist = d if dist is None else torch.cat([dist, d], dim=1)

        # get labels, contact with retained
        candidates = labels.view(1,-1).expand(len(testLabels), -1).to(out_device)
        trainLabels = candidates if trainLabels is None else torch.cat([trainLabels, candidates], dim=1)
        
        # get indexes, contact with retained
        candidate_indexes = indexes.view(1,-1).expand(len(testLabels), -1).to(out_device)
        trainIndexes = candidate_indexes if trainIndexes is None else torch.cat([trainIndexes, candidate_indexes], dim=1)
        
        # keep the top K distances and labels  
        #if batch_num % 10 == 0 or batch_num == (len(train_loader)-1):
        yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
        dist = torch.gather(dist, 1, yi)
        trainLabels = torch.gather(trainLabels, 1, yi)
        trainIndexes = torch.gather(trainIndexes, 1, yi)
        
        trainFeatures, labels, indexes = None, None, None
    
    # generate weighted predictions
    batchSize = len(testLabels)
    retrieval = trainLabels
    retrieval_one_hot = torch.zeros(K, C).to('cpu')
    retrieval_one_hot.resize_(batchSize * K, C).zero_()
    retrieval_one_hot.scatter_(1, retrieval.view(-1, 1).cpu(), 1)
    yd_transform = dist.clone().div_(sigma).exp_().cpu()
    probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
    _, predictions = probs.sort(1, True)
    
    # Find which predictions match the target
    correct = predictions.eq(testLabels.view(-1,1).cpu())
    correct.shape
    
    total = correct.size(0)
    top1 = correct.narrow(1,0,1).sum().item() / total * 100
    top5 = correct.narrow(1,0,5).sum().item() / total * 100
    
    print(f"run_kNN accuracy: top1={top1}, top5={top5}")
    
    return top1, top5

