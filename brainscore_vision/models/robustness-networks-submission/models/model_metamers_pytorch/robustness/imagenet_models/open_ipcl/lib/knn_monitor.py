import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from pdb import set_trace

try:
    from fastprogress.fastprogress import progress_bar
except:
    from fastprogress import progress_bar
    
def knn_monitor(model, trainFeatures, trainLabels, val_loader, 
                K=200, sigma=1.0, num_chunks=200, knn_device='cpu'):
    
    trainFeatures = trainFeatures.to(knn_device, non_blocking=True) 
    trainLabels = trainLabels.to(knn_device, non_blocking=True)
    
    print("extracting test features...")
    testFeatures, testLabels, indexes = get_features(model, val_loader, to_device=knn_device)
    testFeatures = testFeatures.to(knn_device, non_blocking=True)
    testLabels = testLabels.to(knn_device, non_blocking=True)
    
    # make sure all the features are l2-normalized
    trainFeatures = nn.functional.normalize(trainFeatures, p=2, dim=1)
    testFeatures = nn.functional.normalize(testFeatures, p=2, dim=1)

    print("running kNN test...")
    
    # split test features into chunks to avoid out-of-memory error:
    chunkFeatures = torch.chunk(testFeatures, num_chunks, dim=0)
    chunkLabels = torch.chunk(testLabels, num_chunks, dim=0)

    C = trainLabels.max() + 1
    top1, top5, total = 0., 0., 0.
    for features, labels in progress_bar(zip(chunkFeatures, chunkLabels), total=num_chunks):
        top1_, top5_, total_ = do_kNN(trainFeatures, trainLabels, features, labels, C, K, sigma, device=knn_device)
        top1 += top1_ / 100 * total_
        top5 += top5_ / 100 * total_
        total += total_
    top1 = top1 / total * 100
    top5 = top5 / total * 100
    
    print(f"run_kNN accuracy: top1={top1}, top5={top5}")
    
    return top1, top5

def get_features(model, dataloader, device=None, to_device=None):
    features,labels,indexes = [],[],[]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if to_device is None:
        to_device = device
        
    model.to(device)
    model.eval()
    with torch.no_grad():
        for imgs,targs,idxs in progress_bar(dataloader):
            if isinstance(imgs, list):
                out = model(imgs[0].to(device, non_blocking=True))
            else:
                out = model(imgs.to(device, non_blocking=True))
            features.append(out.view(out.shape[0],-1).to(to_device, non_blocking=True))
            if isinstance(targs, list):
                labels.append(targs[0].to(to_device, non_blocking=True))
            else:
                labels.append(targs.to(to_device, non_blocking=True))
            if isinstance(idxs, list):
                indexes.append(idxs[0].to(to_device, non_blocking=True))
            else:
                indexes.append(idxs.to(to_device, non_blocking=True))
    
    try:
        features = torch.cat(features)
        labels = torch.cat(labels)
        indexes = torch.cat(indexes)
    except Exception as e: 
        print(e)        
        set_trace()
    
    return features, labels, indexes

def do_kNN(trainFeatures, trainLabels, testFeatures, testLabels, C, K, sigma, device=None):
    '''
        trainFeatures: [nTrainSamples, nFeatures]
        trainLabels: [nTrainSamples]
        
        testFeatures: [nTestSamples, nFeatures]
        testLabels: [nTestSamples]
    '''
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    batchSize = len(testLabels)
    
    dist = torch.mm(testFeatures, trainFeatures.T).to(device, non_blocking=True)
    
    yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
    
    candidates = trainLabels.view(1,-1).expand(batchSize, -1).to(device, non_blocking=True)
    
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

def repeated_kfold_kNN(model, dataloader, K=40, sigma=.07, n_splits=5, n_repeats=5, random_state=123456, epoch=None):
    print("\n================================")
    print(' Generalization Epoch: %d' % epoch)
    print("================================")
    
    print(f"running repeated k-fold validation (n_splits={n_splits}, n_repeats={n_repeats})")
    print("extracting features...")
    features, labels, indexes = get_features(model, dataloader)
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

def run_kNN(model, train_loader, test_loader, K=200, sigma=.07, num_chunks=200, knn_device=None):
    print("extracting training features...")
    trainFeatures, trainLabels, indexes = get_features(model, train_loader, to_device=knn_device)
    print("extracting test features...")
    testFeatures, testLabels, indexes = get_features(model, test_loader, to_device=knn_device)
    print("running kNN test...")
    
    trainFeatures = nn.functional.normalize(trainFeatures, p=2, dim=1)
    testFeatures = nn.functional.normalize(testFeatures, p=2, dim=1)
    
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