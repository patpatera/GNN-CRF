# OGB datasets
import random
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset

import os.path as osp

import torch
import torch_geometric.transforms as T


#PyG datasets
from torch_geometric.datasets import Planetoid, ZINC, TUDataset, Amazon, Flickr, Reddit, WebKB, Coauthor, MNISTSuperpixels
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops, add_self_loops, coalesce, index_to_mask

from utils.splits import RandomNodeSplit

import numpy as np


class Dataset(torch.utils.data.Dataset):
   def __init__(self):
      pass
   

def load_large_dataset(name, w_text=False, idx2mask=True):
  name = name.lower()

  if name == 'ogbn-arxiv':
    dataset = get_dataset_node_OGB(name, w_text, idx2mask)
  else:
    raise Exception(f"Not supported dataset: {name}")
  
  return dataset

def get_dataset_node_OGB(name, w_text=False, idx2mask=False):
    
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = PygNodePropPredDataset(name=name, root=path)

    dataset.data.edge_index, _ = remove_self_loops(dataset.data.edge_index)
    dataset.data = T.ToUndirected()(dataset.data)
    dataset.data = T.RemoveDuplicatedEdges()(dataset.data)

    #dataset.data = T.RemoveIsolatedNodes()(dataset.data)
    #dataset.data = T.NormalizeFeatures()(dataset.data)
    #dataset.data = T.AddSelfLoops()(dataset.data)
    split_idx = dataset.get_idx_split()    
        
    # Split dataset with corresponing masks
    for split in ['train', 'valid', 'test']:
        if idx2mask:
          dataset.data[f'{split}_mask'] = index_to_mask(split_idx[split], dataset.data.y.shape[0])
        else:
           dataset.data[f'{split}_mask'] = split_idx[split]

    if w_text:
      text_p = osp.join("/media/Patrik/data", name, name.replace("-", "_"), "embeddings/X.all.txt")

      #text = []
      f = open(text_p, "r")
      text = f.readlines()
      text = [s.rstrip() for s in text]
      
      """     
      ftext = []

      for line in text:
          i = 0
          for c in line:
             if c.isupper():
                break
             i += 1

          query = "Give me context to following title: " + line[:i]
          ftext.append(query)
      #    ftext.append({"title": line[:i], "abstract": line[i:]})
      """
      f.close()

      
      dataset.data["text"] = text

    return dataset

def get_dataset_OGB(name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)

    dataset = PygGraphPropPredDataset(name=name, root=path)

    return dataset


def get_dataset_TU(name, node_ft=False, normalize_feat=True, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)

    transforms = __add_transform(normalize_feat, transform)
    dataset = TUDataset(path, name, transforms, use_node_attr=node_ft)

    return dataset


def get_dataset_ZINC(name, split="train", subset=True, normalize_feat=True, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)

    transforms = __add_transform(normalize_feat, transform)
    dataset = ZINC(path, subset, split, transforms)

    return dataset

def get_dataset_planetoid(name, normalize_feat=True, transform=None, split="public", data_seed=42, split_i=0):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
 
    if split == 'complete':
        dataset = Planetoid(path, name)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[:dataset[0].num_nodes - 1000] = 1
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500:] = 1
    
    transforms = __add_transform(True, transform)
    random_split = False #True # True
    use_lcc = False # True

    #dataset = Planetoid(path, name, split=split, transform=transforms)
    #dataset = Planetoid(path, name, split="full")
    nm = name.lower()

    if nm in ['computers', 'photo']:
      dataset = Amazon(path, name)
      random_split = use_lcc = True
    elif nm in ['texas', 'wisconsin', 'cornell']:
       dataset = WebKB(path, name)
       dataset.data = webk_split(path, name, dataset, split=split_i)
       random_split = use_lcc = False
       random_split = False
    elif nm in ['coauthor']:
       dataset = Coauthor(path, "CS")
       random_split = use_lcc = True
    elif nm in ['reddit']:
       dataset = Reddit(path)
       random_split = False
       use_lcc = False
    elif nm in ['flickr']:
       dataset = Flickr(path)
       random_split = False
       use_lcc = False
    elif nm in ['mnists']:
      dataset = MNISTSuperpixels(path, train=True)
    else:
      dataset = Planetoid(path, name, split='public')
      #dataset = Planetoid(path, name, split='random', num_train_per_class=20, num_val=210, num_test=1956)

      #dataset = Planetoid(path, name, split='random', num_train_per_class=20, num_val=210, num_test=19567)

      #tt= dataset.data.train_mask.bitwise_or(dataset.data.test_mask)
      #dataset.data.val_mask = ~tt.clone()
      #if name.lower() == "cora":
      #  dataset = Planetoid(path, name, split='random', num_train_per_class=20, num_val=210, num_test=2135) # Cora
      #elif name.lower() == "pubmed":
      #  dataset = Planetoid(path, name, split='random', num_train_per_class=20, num_val=210, num_test=19567) #Pubmed
      #else:
      #  #dataset = Planetoid(path, name, split='random', num_train_per_class=20, num_val=210, num_test=19567) #Pubmed
    num_cls = dataset.num_classes
    

    #dataset.data = T.NormalizeFeatures()(dataset.data)  # Only for: Photo
    dataset.data.edge_index, _ = remove_self_loops(dataset.data.edge_index)
    
    
    if nm in ['texas', 'wisconsin', 'cornell']:
       dataset.data.edge_index, _ = add_self_loops(dataset.data.edge_index, num_nodes=dataset.data.x.shape[0])
       return dataset

    #dataset.data = T.AddSelfLoops()(dataset.data)
    dataset.data = T.RemoveIsolatedNodes()(dataset.data)
    
    dataset.data = T.ToUndirected()(dataset.data)
    
    num_comp = 1 #random.randint(1, 5)    # 5=Cora, 3=Pubmed/Cite, 
    dataset.data = T.LargestConnectedComponents(num_components=num_comp)(dataset.data)
    #dataset.data = T.GDC( sparsification_kwargs=dict(method='topk', k=12, dim=0, avg_degree=12, eps=1e-2))(dataset.data)
    
    
    dataset.data = T.RemoveDuplicatedEdges()(dataset.data)
    dataset.data.edge_index = coalesce(dataset.data.edge_index, num_nodes=dataset.data.x.shape[0])

  
    val = 30 * num_cls
    dataset.data = T.RandomNodeSplit("test_rest", num_train_per_class=20, num_val=val)(dataset.data)


    #use_lcc = True
    random_split = False

    
    #num = dataset.data.num_nodes

    # 60/20/20 splits (train/val/test) random
    #num_train = int((num*0.6) / dataset.num_classes)
    #num_rest = int(num_train / 2)
    #dataset.data = T.RandomNodeSplit("random", num_train_per_class=num_train, num_val=num_rest, num_test=num_rest)(dataset.data)

    #dataset.data = T.RandomNodeSplit("random", num_train_per_class=20, num_val=500, num_test=1500)(dataset.data)


    dataset.data["valid_mask"] = dataset.data.val_mask
    if False: #if use_lcc:
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

    if random_split:
      print("Dataset: Using random splits protocol...")
      seed = random.randint(0, 1000)
      dataset.data = set_train_val_test_split(
        seed,
        dataset.data,
        num_development=5000 if name == "coauthor" else 1500,
        num_per_class=20)
      
    print("Nodes:\n\tTrain: ", dataset.data.train_mask.sum(), "Test: ", dataset.data.test_mask.sum(), "Val: ", dataset.data.valid_mask.sum())

    return dataset

def webk_split(path, name, dataset, split=0): 
    data = dataset[0]
    splits_file = np.load(f'{path}/{name.lower()}/raw/{name.lower()}_split_0.6_0.2_{split}.npz')

    train_mask = splits_file['train_mask']
    val_mask = splits_file['val_mask']
    test_mask = splits_file['test_mask']

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    return data

def __add_transform(normalize_feat=True, transform=None):
    transforms = []
    if normalize_feat or not transform==None:
        transforms = []

        if normalize_feat:
            transforms.append(T.NormalizeFeatures())
        if not transform == None:
            transforms.append(transform)

        transforms = T.Compose([*transforms])

    return transforms


def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
  
  rnd_state = np.random.RandomState(seed)
  num_nodes = data.y.shape[0]
  development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
  test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

  train_idx = []
  rnd_state = np.random.RandomState(seed)
  for c in range(data.y.max() + 1):
    class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
    train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

  val_idx = [i for i in development_idx if i not in train_idx]

  def get_mask(idx):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

  data.train_mask = get_mask(train_idx)
  data.valid_mask = get_mask(val_idx)
  data.test_mask = get_mask(test_idx)

  return data 


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = dataset.data.edge_index.numpy()
  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes

def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
  remaining_nodes = set(range(dataset.data.x.shape[0]))
  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(dataset, start)
    comps.append(comp)
    remaining_nodes = remaining_nodes.difference(comp)
  return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 1
  return mapper


def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]



if __name__ == "__main__":
    print("Loading dataset...")

    from torch_geometric.loader import DataLoader
    from tqdm import tqdm

    name = "ogbn-products"
    bs = 64

    from ogb.nodeproppred import PygNodePropPredDataset
    from utils.graph_utils import index_to_mask, extract_adj_mat
    import torch.nn.functional as F

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = PygNodePropPredDataset(name, path)

    split_idx = dataset.get_idx_split()
    data = dataset[0]

    a = torch.rand((200, 5)).argmax(-1)
    print("A: ", a.shape, a)

    f = F.one_hot(a, num_classes=5)
    print(f)

 
    #split_idx = dataset.get_idx_split()
    #train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    #graph = dataset[0] # pyg graph object

    #train_loader = NeighborLoader(graph, input_nodes=train_idx,
    #                           num_neighbors=[5], batch_size=1024,
    #                           shuffle=True, num_workers=12)
    

    #total = 0
    #for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
    #    print("Step: ", step)
    #    print("Data: ", batch.train_mask)
    #    total += batch.x.shape[0]
        #print("Batch: ", batch)
    #    print("===============")

    print("Finished...")


    """
    dataset = get_dataset_TU("DD", True)
    train_loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=8)
    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        print("Data: ", batch.x.shape, " => ", batch.x)
        print("Labels: ", batch.y.shape)
        print(batch)
        print("===============")
    
    print("# labels: ", dataset.num_node_features, dataset.num_node_labels)
    print(dataset.data.x[0].shape)
    for i in range(dataset.data.x[0].shape[0]):
        print("Id: ", (i+1), "-->", dataset.data.x[2][i].item())
    """

    #dataset = get_dataset_OGB("ogbg-molpcba")
    #split_idx = dataset.get_idx_split()

    #train_loader = DataLoader(dataset[split_idx["train"]], batch_size=bs, shuffle=True, num_workers=8)
    
    #for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
    #    print("Data: ", batch.x.shape, " => ", batch.x)
    #    print(batch)
    #    print("===============")

    print("Creating visualisation...")
    #GraphVis.vis(data)