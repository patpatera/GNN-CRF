#Copy from https://github.com/pyg-team/pytorch_geometric/blob/081a964638459d1374f2e90de8ccfc9f0ea1f043/torch_geometric/utils/mask.py#L7

import numpy as np
import torch

# PyG packages
from torch_geometric.utils import to_dense_adj, remove_self_loops, degree, to_undirected
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from torch_cluster import knn_graph, radius_graph
from torch_sparse import coalesce, spspmm

from scipy.sparse import coo_matrix
import numba

import pynndescent

# Loaders
from torch_geometric.loader import ClusterLoader, ClusterData

import torch.nn.functional as F
import torch_geometric.transforms as T


from sklearn.neighbors import NearestNeighbors, KDTree


# Copied from GRAND (Graph Neural Diffusion)
def apply_feat_KNN(data, x, k, edges):
    D = torch_geometric.utils.degree(data.edge_index[1], num_nodes=x.shape[0]).cpu()
    D_mask = D==0

    x = x.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(x)
    distances, indices = nbrs.kneighbors(x)

    arr = torch.arange(0, x.shape[0])[D_mask].cuda()

    src = np.linspace(0, len(x) * k, len(x) * k + 1)[:-1] // k
    dst = indices.reshape(-1)
    AA = np.vstack((src, dst))
    AA = torch.from_numpy(AA).long().to(data.edge_index.device)
    AA = to_undirected(AA)


    mask = torch.isin(AA[0], arr).bitwise_or(torch.isin(AA[1], arr))
    AA = AA[:, mask]
    

    A = torch.cat([data.edge_index, AA], dim=1)
    A = torch_geometric.utils.coalesce(A)
    A, _ = torch_geometric.utils.remove_self_loops(A)
    #A, _ = torch_geometric.utils.dropout_edge(A, p=0.2, force_undirected=True)
    


    data.edge_index = A
    data = T.RemoveDuplicatedEdges()(data)
    #data = T.AddSelfLoops()(data)
    A = data.edge_index

    n_nodes, n_edges = x.shape[0], A.shape[1]
    s, t = A[0], A[1]
    edge_dict = torch.sparse_coo_tensor(indices=A, values=torch.arange(n_edges).cuda(), size=(n_nodes, n_nodes)).to_dense()
    data["rev"] = edge_dict[t, s]

    return data


def knn_graph_rewire(data, x):
    AA = knn_graph(x, 4, cosine=True, num_workers=12).cuda()
    AA = to_undirected(AA)

    A = torch.cat([data.edge_index, AA], dim=1)
    A = torch_geometric.utils.coalesce(A)

    data.edge_index = A 

    n_nodes, n_edges = x.shape[0], A.shape[1]
    s, t = A[0], A[1]
    edge_dict = torch.sparse_coo_tensor(indices=A, values=torch.arange(n_edges).cuda(), size=(n_nodes, n_nodes)).to_dense()
    data["rev"] = edge_dict[t, s]

    return data


def sharpen(prob, temp):
    prob_pow =  torch.pow(prob, (1.0 / temp))
    row_sum = prob_pow.sum(dim=1).reshape(-1, 1)
    
    return (prob_pow / row_sum)


def get_l_hops(A, l):
    """
        Extract l-hops from dense adjacency matrix.
        Impl. from StructPool: https://github.com/Nate1874/StructPool/blob/master/pool.py

        Params:
            A (Tensor): Dense adjacency matrix [NxN].
            l (int): The number of hops.
    """
    if l == 1:
        return A

    A_l = A
    previous = A

    for _ in range(1, l):
        now = torch.matmul(previous, A)
        A_l = now  + A_l
        previous = now

    # Add self-loops
    A_l[range(A.size()[0]), range(A.size()[0])] = 1
    
    # Convert to [0;1] -- adjacency representation
    A_l[A_l>0] = 1.

    return A_l

def get_isolated_mask(A):
    """
        Get boolean mask of isolated nodes.
        Params:
            A (Tensor): Dense adjacency matrix.
    """
    iso_mask = ((A.sum(-1)<1) & (A.sum(0)<1)).cuda() # Get mask for isolated nodes
    return iso_mask


def get_adj_by_pred(y, scores=None, mask=None, i=0, gt=None):
    y_ = y.squeeze(1).repeat(y.shape[0], 1)

    # Select nodes with certain score
    # increasing the score during MFI iteration.
    if not scores==None:
        q = 0.7 + 0.5*i
        y[scores<q] = -1

    # Mask some nodes w.r.t. mask
    if not mask==None:
        y[mask] = -1

    y_ = torch.where(y_==y, 1., 0.)

    # Compare with ground truth labels -- for testing purpose
    if not gt==None:
        t_ = gt.squeeze(1).repeat(gt.shape[0], 1)
        t_ = torch.where(t_==gt, 1., 0.)
        num = (t_ * y_).bool().sum()
        print("\tCorrect: ", (num/t_.bool().sum()).item(), "From: ", y_.sum().item())


    y_ = y_ + y_.t()
    y_[y_>0] = 1.

    return y_


def l2_norm(x, eps = 1e-20):
    norm = x.norm(dim = 1, keepdim = True).clamp(min = eps)
    return x / norm

def get_adj_by_cossim( x, k=20, self_conn=False):
    """
        Create adjacency by cosine similarity of node features
        by the top-k most similar nodes

        Params:
            x (Tensor): node features.
            k (int): top-k hyper-parameter.
            self_conn (bool): if add self-loop connections.

    """
    attn = l2_norm(x)
    attn = torch.mm(attn, attn.T)

    _, top = attn.topk(k, dim=-1)
    mask = torch.zeros_like(attn).cuda()

    attn = mask.scatter_(1, top, 1.)

    if self_conn:
        attn = attn + attn.T
    
    attn[attn>0] = 1.

    return attn


def nxn_cos_sim(A, B, dim=1, eps=1e-8):
    """
        Different version of 'get_adj_by_cossim'.
        For testing purpose if cos-sim works.
    """
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)

# N = data.num_nodes
def add_two_hop(edge_index, N, edge_attr=None):
    value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

    index, value = spspmm(edge_index, value, edge_index, value, N, N, N)
    value.fill_(0)
    index, value = remove_self_loops(index, value)

    edge_index = torch.cat([edge_index, index], dim=1)
    if edge_attr is None:
        edge_index, _ = coalesce(edge_index, None, N, N)
    else:
        value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
        value = value.expand(-1, *list(edge_attr.size())[1:])
        edge_attr = torch.cat([edge_attr, value], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
        edge_attr = edge_attr

    return edge_index

def softA(A, attn, neg_lam=10.):
    pos_a = (A==0).logical_and(attn>0)
    pos_n = (A==0).logical_and(attn<=0.)

    attn[pos_a] = torch.log(attn[pos_a])
    attn[pos_n] = attn[pos_n] * neg_lam

    return attn

def create_knn(embed, num_edges=3):
    # Clustering
    from torch_cluster import knn_graph, radius_graph
    return knn_graph(embed.cuda(), num_edges, cosine=True, num_workers=12)

def extract_adj_mat(edge_index, num_nodes, edge_attr=None, to_undirected=False, add_self_loops=False, embed=None):
    # Convert sparse adjacency to dense adjecancy matrix (NxN)!
    #if edge_index.numel() <= 0:
    #    edge_index = create_knn(embed)

    if edge_index.numel() <= 0:
        A = torch.zeros((num_nodes, num_nodes), device=edge_index.device).float()
    elif num_nodes>0:
        A = to_dense_adj(edge_index, edge_attr=edge_attr, max_num_nodes=num_nodes).cuda().squeeze(0)
    else:
        A = to_dense_adj(edge_index, edge_attr=edge_attr).cuda().squeeze(0)

    #assert (A>0.).sum()==edge_index.shape[1]

    mask_border = None
    mask_border = A.sum(-1) <= 0

    #if mask_border.sum() > 0:
    #    edge_index_a = create_knn(embed)
    #    B = to_dense_adj(edge_index_a, edge_attr=edge_attr, max_num_nodes=num_nodes).cuda().squeeze(0)
    #    A[mask_border] = A[mask_border] + B[mask_border]

    if to_undirected:
        A_T = A.T.clone()
    #    #A_T[A>0.] = 0. 
        A = A + A_T
    
    if add_self_loops:
    #    # Add self-loop only to nodes without any connection!
        A[mask_border, mask_border] = 1.
    #    #A = A + torch.eye(num_nodes, device=A.device).float()

    if to_undirected or add_self_loops:
        A[A>0.] = 1.

    #if to_bool_neg:
    #    A = ~(A.bool())

    return A, mask_border


def create_cluster_loader(path, data, embed, args, rebuild_adj=False, num_neigh=20):
    if rebuild_adj:
        data.edge_index = build_clusters_pynndescent(embed, n_neigh=30)
        #print("Computing KNN-Graph by PyG...")
        #data.edge_index = knn_graph(embed, num_neigh, cosine=True, num_workers=12).detach().cpu()

    path = "/media/patpa/f0c432b3-57ba-4522-a804-cab376ec28835/projects/data/prod_clusters/"

    if "arxiv" in args.dataset:
        path = None # No need to save clusters -- creating is fast for ogbn-arxiv 

        # Need to transform to undirected graph here of METIS calculation will failed on Segmentation fault!
        from torch_geometric.transforms import ToUndirected
        # This undirected class works correctly! Tested...
        data = ToUndirected()(data)


    cluster_data = ClusterData(data, num_parts=args.num_clusters, recursive=False, save_dir=path)
    train_loader = ClusterLoader(cluster_data, batch_size=args.bs, shuffle=True, num_workers=args.j)

    cluster_data = ClusterData(data, num_parts=args.num_clusters_test, recursive=False, save_dir=path)
    test_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False, num_workers=args.j)

    return train_loader, test_loader

@numba.njit(fastmath=True)
def correct_alternative_cosine(ds):
    result = np.empty_like(ds)
    for i in range(ds.shape[0]):
        result[i] = 1.0 - np.power(2.0, ds[i])
    return result

def build_clusters_pynndescent(embed, A_l, n_neigh=20, p=-1):
    print("Calculating KNN-Graph by pynndescent...")

    pynn_dist_fns_fda = pynndescent.distances.fast_distance_alternatives
    pynn_dist_fns_fda["cosine"]["correction"] = correct_alternative_cosine
    pynn_dist_fns_fda["dot"]["correction"] = correct_alternative_cosine      

    index = pynndescent.NNDescent(embed, metric="cosine", n_neighbors=n_neigh, n_jobs=16)  # it takes a while to build index!
    adj, dist = index.neighbor_graph
    print("\tFinished...")

    import os
    
    #path = os.path.join("/media/Patrik/data", args.dataset, args.dataset.replace("-", "_"), "embeddings/pydes_knn_xrt_10.pt")

    # use 'scipy.sparse.coo.coo_matrix' to replace  'torch.sparse_coo_tensor'
    N, D = adj.shape
    
    # build a sparse matrix: sparse.coo_matrix((V,(I,J)),shape=(N,N))
    I = np.arange(N).repeat(D)
    J = adj.flatten()

    # filter by distance threshold
    p =-1
    if p >= 0:
        cutoff = p
        cutoff = np.percentile(dist.flatten(), p)
        chosen = (dist.flatten() <= cutoff)
        I = I[chosen]
        J = J[chosen]

    V = np.ones(I.shape[0])

    index = (I, J)
    A_knn = coo_matrix((V, index), shape=(N, N))

    #A_l = to_scipy_sparse_matrix(A_l, num_nodes=embed.shape[0]).tocoo()
    #A_knn = (A_knn - A_l).tocoo()     # Remove local edges from global
    #del A_l
    
    mask = A_knn.data > 0   # Filter out dges
    A_knn.data = A_knn.data[mask]
    A_knn.row = A_knn.row[mask]
    A_knn.col = A_knn.col[mask]

    # return data.edge_index structure
    return from_scipy_sparse_matrix(A_knn)[0]


def get_comp_graph_pdf(var, params):
    from graphviz import Digraph
    """ Produces Graphviz representation of PyTorch autograd graph
    
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)
    
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    
    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

def remap(src, edges):
    map_src = {}
    i = 0
    for x in src:
        if not x.item() in map_src:
            map_src[x.item()] = i
            i += 1
    return torch.tensor([map_src[x.item()] for x in edges])

def extract_global(edge_index, x, glob_idx):
    # intra_glob == global nodes as source in connections => updating nodes in the cluster!
    edge_index = edge_index.cuda()

    g_mask = torch.isin(edge_index[0], glob_idx)
    # Filter connection withing clusert in global context
    #intra = torch.isin(edge_index[1], glob_idx).logical_and(g_mask)
    #g_mask = g_mask.logical_xor(intra)

    dest_nodes = torch.unique(edge_index[1, g_mask])

    src_edges = remap(glob_idx, edge_index[0, g_mask])
    des_edges = remap(dest_nodes, edge_index[1, g_mask])      

    edge_index = torch.stack([src_edges, des_edges], dim=0)       

    return x[dest_nodes], edge_index, dest_nodes

def extract_global_outer(edge_index, x, glob_idx):
    edge_index = edge_index.cuda()

    g_mask = torch.isin(edge_index[1], glob_idx)
    outer_nodes = torch.unique(edge_index[0, g_mask])   

    cluster_edges = remap(glob_idx, edge_index[1, g_mask])
    outer_edges = remap(outer_nodes, edge_index[0, g_mask])      

    edge_index = torch.stack([outer_edges, cluster_edges], dim=0)        

    return x[outer_nodes].cuda(), edge_index.cuda(), outer_nodes.cuda()

def get_border_nodes_mask(edge_index, num_nodes, th_degree=2):
    d_nodes = degree(edge_index[0], num_nodes)
    d_nodes = d_nodes[edge_index[0]]

    return d_nodes

def mixup(data, k=1):
    Ag = torch.zeros((data.x.shape[0], data.x.shape[0])).cuda()
    r, c = data.edge_index
    Ag[r, c] = 1.
    Ag[range(Ag.shape[0]), range(Ag.shape[1])] = 1.

    idx = torch.multinomial(Ag, k, replacement=True)
    #Ag_ = torch.zeros_like(Ag).cuda()
    #Ag  = Ag_.scatter(1, idx, 1.)

    return idx

def rewire_attn(attn):
    attn[range(attn.shape[0]), range(attn.shape[0])] = -100
    top_idx = torch.topk(attn, 2, -1)[1]

    adj = torch.zeros_like(attn).cuda()
    adj = adj.scatter(1, top_idx, 1.)

    return adj 

def fake_labels(data, attn):
    train_lab = data.y.clone()
    y = torch.stack([train_lab, train_lab], dim=1).cuda()

    y_fake = -1 * torch.ones_like(data.y).cuda()
    top_idx = torch.topk(attn, 5, -1)[1]
    
    y_fake[top_idx[data.train_mask]] = y[data.train_mask]
    mask = (y_fake > -1)

    return y_fake, mask


def rand_global_edges(data, k=5):
    # Extract global dense adjacency matrix [NxM]
    Ag = torch.zeros((data.x.shape[0], data.g_x.shape[0])).cuda()
    r, c = data.g_edge_index
    Ag[r, c] = 1.
    
    # Select ranodm edges in global connections
    if k > 0:
        idx = Ag.multinomial(k, replacement=True)
        Ag_ = torch.zeros_like(Ag).cuda()
        Ag = Ag_.scatter(1, idx, 1.)

    return Ag

def rand_global_edge_outers(data, k=5):
    # Extract global dense adjacency matrix [NxM]
    data["A_g"] = torch.zeros((data.g_x.shape[0], data.x.shape[0])).cuda()
    r, c = data.g_edge_index
    data.A_g[r, c] = 1.
    
    # Select ranodm edges in global connections
    if k > 0:
        idx = data.A_g.multinomial(k, replacement=True)
        Ag = torch.zeros_like(data.A_g).cuda()
        data.A_g = Ag.scatter(1, idx, 1.)

    return data

def comp_to_edges(comp, idx):
    _, idx = torch.max(idx, dim=1)

    M = comp.T[idx][:, idx]
    #m1 = torch.where(M>0.01, 1., 0.).bool().cuda()
    #m2 = torch.where(M<-0.82, 1., 0.).bool().cuda()
    
    return M


def sinkhorn(pred, eta=3):
    pred = F.softmax(pred, dim=-1) + 1e-20
    PS = pred
    N, K = PS.shape[0], PS.shape[1]

    PS = PS.T
    c = torch.ones((N, 1)) / N
    r = torch.ones((K, 1)) / K

    # = r_in.cuda()
    c = c.cuda()
    r = r.cuda()

    # average column mean 1/N
    PS = torch.pow(PS, eta)+1e-20  # K x N
    #r_init = copy.deepcopy(r)

    inv_N = 1. / N
    inv_K = 1. / K

    err = 1e6
    # error rate
    _counter = 1
    for _ in range(50):
        if err < 1e-1:
            break

        r = inv_K / (PS @ c)          # (KxN)@(N,1) = K x 1
        c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1

        if _counter % 10 == 0:
            err = torch.nansum(torch.abs(c / c_new - 1))

        c = c_new
        _counter += 1

    PS = PS * torch.squeeze(c)
    PS = PS.T
    PS = PS * torch.squeeze(r)
    #PS = PS.T

    return PS.max(-1)

import warnings

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.typing import Adj, SparseTensor
from torch_geometric.utils import scatter
import torch_sparse

def is_torch_sparse_tensor(src) -> bool:
    r"""Returns :obj:`True` if the input :obj:`src` is a
    :class:`torch.sparse.Tensor` (in any sparse layout).

    Args:
        src (Any): The input object to be checked.
    """
    if isinstance(src, Tensor):
        if src.layout == torch.sparse_coo:
            return True
        if src.layout == torch.sparse_csr:
            return True

    return False

@torch.jit._overload
def spmm(src, other, reduce):
    # type: (Tensor, Tensor, str) -> Tensor
    pass


@torch.jit._overload
def spmm(src, other, reduce):
    # type: (SparseTensor, Tensor, str) -> Tensor
    pass

def spmm(src: Adj, other: Tensor, reduce: str = "sum") -> Tensor:
    """Matrix product of sparse matrix with dense matrix.

    Args:
        src (Tensor or torch_sparse.SparseTensor): The input sparse matrix,
            either a :pyg:`PyG` :class:`torch_sparse.SparseTensor` or a
            :pytorch:`PyTorch` :class:`torch.sparse.Tensor`.
        other (Tensor): The input dense matrix.
        reduce (str, optional): The reduce operation to use
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`
    """
    reduce = 'sum' if reduce == 'add' else reduce

    if reduce not in ['sum', 'mean', 'min', 'max']:
        raise ValueError(f"`reduce` argument '{reduce}' not supported")

    if isinstance(src, SparseTensor):
        if (torch_geometric.typing.WITH_PT2 and other.dim() == 2
                and not src.is_cuda()):
            # Use optimized PyTorch `torch.sparse.mm` path:
            csr = src.to_torch_sparse_csr_tensor()
            return torch.sparse.mm(csr, other, reduce)
        return torch_sparse.matmul(src, other, reduce)

    if not is_torch_sparse_tensor(src):
        raise ValueError("`src` must be a `torch_sparse.SparseTensor` "
                         f"or a `torch.sparse.Tensor` (got {type(src)}).")

    # `torch.sparse.mm` only supports reductions on CPU for PyTorch>=2.0.
    # This will currently throw on error for CUDA tensors.
    if torch_geometric.typing.WITH_PT2:

        if src.is_cuda and (reduce == 'min' or reduce == 'max'):
            raise NotImplementedError(f"`{reduce}` reduction is not yet "
                                      f"supported for 'torch.sparse.Tensor' "
                                      f"on device '{src.device}'")

        # Always convert COO to CSR for more efficient processing:
        if src.layout == torch.sparse_coo:
            warnings.warn(f"Converting sparse tensor to CSR format for more "
                          f"efficient processing. Consider converting your "
                          f"sparse tensor to CSR format beforehand to avoid "
                          f"repeated conversion (got '{src.layout}')")
            src = src.to_sparse_csr()

        # Warn in case of CSC format without gradient computation:
        if src.layout == torch.sparse_csc and not other.requires_grad:
            warnings.warn(f"Converting sparse tensor to CSR format for more "
                          f"efficient processing. Consider converting your "
                          f"sparse tensor to CSR format beforehand to avoid "
                          f"repeated conversion (got '{src.layout}')")

        # Use the default code path for `sum` reduction (works on CPU/GPU):
        if reduce == 'sum':
            return torch.sparse.mm(src, other)

        # Use the default code path with custom reduction (works on CPU):
        if src.layout == torch.sparse_csr and not src.is_cuda:
            return torch.sparse.mm(src, other, reduce)

        # Simulate `mean` reduction by dividing by degree:
        if reduce == 'mean':
            if src.layout == torch.sparse_csr:
                ptr = src.crow_indices()
                deg = ptr[1:] - ptr[:-1]
            else:
                assert src.layout == torch.sparse_csc
                deg = scatter(torch.ones_like(src.values()), src.row_indices(),
                              dim=0, dim_size=src.size(0), reduce='sum')

            return torch.sparse.mm(src, other) / deg.view(-1, 1).clamp_(min=1)

        # TODO The `torch.sparse.mm` code path with the `reduce` argument does
        # not yet support CSC :(
        if src.layout == torch.sparse_csc:
            warnings.warn(f"Converting sparse tensor to CSR format for more "
                          f"efficient processing. Consider converting your "
                          f"sparse tensor to CSR format beforehand to avoid "
                          f"repeated conversion (got '{src.layout}')")
            src = src.to_sparse_csr()

        return torch.sparse.mm(src, other, reduce)

    # pragma: no cover
    # PyTorch < 2.0 only supports sparse COO format:
    if reduce == 'sum':
        return torch.sparse.mm(src, other)
    elif reduce == 'mean':
        if src.layout == torch.sparse_csr:
            ptr = src.crow_indices()
            deg = ptr[1:] - ptr[:-1]
        elif src.layout == torch.sparse_csc:
            assert src.layout == torch.sparse_csc
            deg = scatter(torch.ones_like(src.values()), src.row_indices(),
                          dim=0, dim_size=src.size(0), reduce='sum')
        else:
            assert src.layout == torch.sparse_coo
            src = src.coalesce()
            deg = scatter(torch.ones_like(src.values()),
                          src.indices()[0], dim=0, dim_size=src.size(0),
                          reduce='sum')

        return torch.sparse.mm(src, other) / deg.view(-1, 1).clamp_(min=1)

    raise ValueError(f"`{reduce}` reduction is not supported for "
                     f"'torch.sparse.Tensor' on device '{src.device}'")
