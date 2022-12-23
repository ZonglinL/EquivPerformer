from utils.utils_profiling import *  # load before other local modules

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import nullcontext

from typing import Dict
from equivariant_attention.kernelization import *

from equivariant_attention.from_se3cnn import utils_steerable
from equivariant_attention.fibers import Fiber, fiber2head
from utils.utils_logging import log_gradient_norm

import dgl
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling

from packaging import version


@profile
def get_basis(G, max_degree, compute_gradients):
    """Precompute the SE(3)-equivariant weight basis, W_J^lk(x)

    This is called by get_basis_and_r().

    Args:
        G: DGL graph instance of type dgl.DGLGraph
        max_degree: non-negative int for degree of highest feature type
        compute_gradients: boolean, whether to compute gradients during basis construction
    Returns:
        dict of equivariant bases. Keys are in the form 'd_in,d_out'. Values are
        tensors of shape (batch_size, 1, 2*d_out+1, 1, 2*d_in+1, number_of_bases)
        where the 1's will later be broadcast to the number of output and input
        channels
    """
    if compute_gradients:
        context = nullcontext()
    else:
        context = torch.no_grad()
    batch_size = G.batch_size
    num_nodes = G.num_nodes()//batch_size
    num_edges = G.num_edges()//(batch_size*num_nodes)
    with context:
        cloned_d = torch.clone(G.edata['d'])

        if G.edata['d'].requires_grad:
            cloned_d.requires_grad_()
            log_gradient_norm(cloned_d, 'Basis computation flow')





        #print(cloned_d.view(batch_size,num_nodes,num_edges,-1).shape)
        cloned_d = cloned_d.view(batch_size,num_nodes,num_edges,-1) ## reshape to BxNxEx3
        cloned_d = cloned_d[:,:,0,:]##BxNx3
        cloned_d = cloned_d.view(batch_size*num_nodes,-1)#back to original shape
        #print(f'dist data is {cloned_d.shape}')
        ##identical across edges
        # Relative positional encodings (vector)
        r_ij = utils_steerable.get_spherical_from_cartesian_torch(cloned_d)##reshape to 20x3
        # Spherical harmonic basis
        Y = utils_steerable.precompute_sh(r_ij, 2*max_degree)
        device = Y[0].device

        '''
        for idx,y in Y.items():
            print(y.view(-1,num_nodes,num_nodes-1,2*int(idx)+1).shape)
        ##confirm identical across edges
        '''

        basis = {}
        for d_in in range(max_degree+1):
            for d_out in range(max_degree+1):
                K_Js = []
                for J in range(abs(d_in-d_out), d_in+d_out+1):
                    # Get spherical harmonic projection matrices
                    Q_J = utils_steerable._basis_transformation_Q_J(J, d_in, d_out)
                    Q_J = Q_J.float().T.to(device)

                    # Create kernel from spherical harmonics
                    K_J = torch.matmul(Y[J], Q_J)
                    K_Js.append(K_J)

                # Reshape so can take linear combinations with a dot product
                size = (-1, 1, 2*d_out+1, 1, 2*d_in+1, 2*min(d_in, d_out)+1)
                basis[f'{d_in},{d_out}'] = torch.stack(K_Js, -1).view(*size)
        return basis


def get_r(G):
    """Compute internodal distances"""
    cloned_d = torch.clone(G.edata['d'])

    if G.edata['d'].requires_grad:
        cloned_d.requires_grad_()
        log_gradient_norm(cloned_d, 'Neural networks flow')

    return torch.sqrt(torch.sum(cloned_d**2, -1, keepdim=True))


def get_basis_and_r(G, max_degree, compute_gradients=False):
    """Return equivariant weight basis (basis) and internodal distances (r).

    Call this function *once* at the start of each forward pass of the model.
    It computes the equivariant weight basis, W_J^lk(x), and internodal
    distances, needed to compute varphi_J^lk(x), of eqn 8 of
    https://arxiv.org/pdf/2006.10503.pdf. The return values of this function
    can be shared as input across all SE(3)-Transformer layers in a model.

    Args:
        G: DGL graph instance of type dgl.DGLGraph()
        max_degree: non-negative int for degree of highest feature-type
        compute_gradients: controls whether to compute gradients during basis construction
    Returns:
        dict of equivariant bases, keys are in form '<d_in><d_out>'
        vector of relative distances, ordered according to edge ordering of G
    """
    basis = get_basis(G, max_degree, compute_gradients)
    r = get_r(G)
    return basis, r


### SE(3) equivariant operations on graphs in DGL

class GConvSE3(nn.Module):
    """A tensor field network layer as a DGL module.

    GConvSE3 stands for a Graph Convolution SE(3)-equivariant layer. It is the
    equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph
    conv layer in a GCN.

    At each node, the activations are split into different "feature types",
    indexed by the SE(3) representation type: non-negative integers 0, 1, 2, ..
    """
    def __init__(self, f_in, f_out, self_interaction: bool=False, edge_dim: int=0, flavor='skip'):
        """SE(3)-equivariant Graph Conv Layer

        Args:
            f_in: list of tuples [(multiplicities, type),...]
            f_out: list of tuples [(multiplicities, type),...]
            self_interaction: include self-interaction in convolution
            edge_dim: number of dimensions for edge embedding
            flavor: allows ['TFN', 'skip'], where 'skip' adds a skip connection
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.edge_dim = edge_dim
        self.self_interaction = self_interaction
        self.flavor = flavor

        # Neighbor -> center weights
        self.kernel_unary = nn.ModuleDict()
        for (mi, di) in self.f_in.structure:
            for (mo, do) in self.f_out.structure:
                self.kernel_unary[f'({di},{do})'] = PairwiseConv(di, mi, do, mo, edge_dim=edge_dim)

        # Center -> center weights
        self.kernel_self = nn.ParameterDict()
        if self_interaction:
            assert self.flavor in ['TFN', 'skip']
            if self.flavor == 'TFN':
                for m_out, d_out in self.f_out.structure:
                    W = nn.Parameter(torch.randn(1, m_out, m_out) / np.sqrt(m_out))
                    self.kernel_self[f'{d_out}'] = W
            elif self.flavor == 'skip':
                for m_in, d_in in self.f_in.structure:
                    if d_in in self.f_out.degrees:
                        m_out = self.f_out.structure_dict[d_in]
                        W = nn.Parameter(torch.randn(1, m_out, m_in) / np.sqrt(m_in))
                        self.kernel_self[f'{d_in}'] = W



    def __repr__(self):
        return f'GConvSE3(structure={self.f_out}, self_interaction={self.self_interaction})'


    def udf_u_mul_e(self, d_out):
        """Compute the convolution for a single output feature type.

        This function is set up as a User Defined Function in DGL.

        Args:
            d_out: output feature type
        Returns:
            edge -> node function handle
        """
        def fnc(edges):
            # Neighbor -> center messages
            msg = 0
            for m_in, d_in in self.f_in.structure:
                src = edges.src[f'{d_in}'].view(-1, m_in*(2*d_in+1), 1)
                edge = edges.data[f'({d_in},{d_out})']
                msg = msg + torch.matmul(edge, src)
            msg = msg.view(msg.shape[0], -1, 2*d_out+1)

            # Center -> center messages
            if self.self_interaction:
                if f'{d_out}' in self.kernel_self.keys():
                    if self.flavor == 'TFN':
                        W = self.kernel_self[f'{d_out}']
                        msg = torch.matmul(W, msg)
                    if self.flavor == 'skip':
                        dst = edges.dst[f'{d_out}']
                        W = self.kernel_self[f'{d_out}']
                        msg = msg + torch.matmul(W, dst)

            return {'msg': msg.view(msg.shape[0], -1, 2*d_out+1)}
        return fnc

    @profile
    def forward(self, h, G=None, r=None, basis=None, **kwargs):
        """Forward pass of the linear layer

        Args:
            G: minibatch of (homo)graphs
            h: dict of features
            r: inter-atomic distances
            basis: pre-computed Q * Y
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """
        with G.local_scope():
            # Add node features to local graph scope
            for k, v in h.items():
                G.ndata[k] = v

            # Add edge features
            if 'w' in G.edata.keys():
                w = G.edata['w']

                feat = torch.cat([w, r], -1)
                #feat = w*r
            else:
                feat = torch.cat([r, ], -1)

            for (mi, di) in self.f_in.structure:
                for (mo, do) in self.f_out.structure:
                    etype = f'({di},{do})'
                    G.edata[etype] = self.kernel_unary[etype](feat, basis)

            # Perform message-passing for each output feature type
            for d in self.f_out.degrees:
                G.update_all(self.udf_u_mul_e(d), fn.mean('msg', f'out{d}'))

            return {f'{d}': G.ndata[f'out{d}'] for d in self.f_out.degrees}


class RadialFunc(nn.Module):
    """NN parameterized radial profile function."""
    def __init__(self, num_freq, in_dim, out_dim, edge_dim: int=0):
        """NN parameterized radial profile function.

        Args:
            num_freq: number of output frequencies
            in_dim: multiplicity of input (num input channels)
            out_dim: multiplicity of output (num output channels)
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        self.num_freq = num_freq
        self.in_dim = in_dim
        self.mid_dim = 32
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        self.net = nn.Sequential(nn.Linear(self.edge_dim+1,self.mid_dim),
                                 BN(self.mid_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.mid_dim,self.mid_dim),
                                 BN(self.mid_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.mid_dim,self.num_freq*in_dim*out_dim))

        nn.init.kaiming_uniform_(self.net[0].weight)
        nn.init.kaiming_uniform_(self.net[3].weight)
        nn.init.kaiming_uniform_(self.net[6].weight)

    def __repr__(self):
        return f"RadialFunc(edge_dim={self.edge_dim}, in_dim={self.in_dim}, out_dim={self.out_dim})"

    def forward(self, x):
        y = self.net(x)
        return y.view(-1, self.out_dim, 1, self.in_dim, 1, self.num_freq)


class PairwiseConv(nn.Module):
    """SE(3)-equivariant convolution between two single-type features"""
    def __init__(self, degree_in: int, nc_in: int, degree_out: int,
                 nc_out: int, edge_dim: int=0):
        """SE(3)-equivariant convolution between a pair of feature types.

        This layer performs a convolution from nc_in features of type degree_in
        to nc_out features of type degree_out.

        Args:
            degree_in: degree of input fiber
            nc_in: number of channels on input
            degree_out: degree of out order
            nc_out: number of channels on output
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        # Log settings
        self.degree_in = degree_in
        self.degree_out = degree_out
        self.nc_in = nc_in
        self.nc_out = nc_out

        # Functions of the degree
        self.num_freq = 2*min(degree_in, degree_out) + 1
        self.d_out = 2*degree_out + 1
        self.edge_dim = edge_dim

        # Radial profile function
        self.rp = RadialFunc(self.num_freq, nc_in, nc_out, self.edge_dim)

    @profile
    def forward(self, feat, basis):
        # Get radial weights
        R = self.rp(feat)
        kernel = torch.sum(R * basis[f'{self.degree_in},{self.degree_out}'], -1)

        return kernel.view(kernel.shape[0], self.d_out*self.nc_out, -1)

class G1x1SE3(nn.Module):
    """Graph Linear SE(3)-equivariant layer, equivalent to a 1x1 convolution.

    This is equivalent to a self-interaction layer in TensorField Networks.
    """
    def __init__(self, f_in, f_out, learnable=True):
        """SE(3)-equivariant 1x1 convolution.

        Args:
            f_in: input Fiber() of feature multiplicities and types
            f_out: output Fiber() of feature multiplicities and types
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out

        # Linear mappings: 1 per output feature type
        self.transform = nn.ParameterDict()
        for m_out, d_out in self.f_out.structure:
            m_in = self.f_in.structure_dict[d_out]
            self.transform[str(d_out)] = nn.Parameter(torch.randn(m_out, m_in) / np.sqrt(m_in), requires_grad=learnable)

    def __repr__(self):
         return f"G1x1SE3(structure={self.f_out})"

    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            if str(k) in self.transform.keys():
                output[k] = torch.matmul(self.transform[str(k)], v)
        return output


class GNormBias(nn.Module):
    """Norm-based SE(3)-equivariant nonlinearity with only learned biases."""

    def __init__(self, fiber, nonlin=nn.ReLU(inplace=True),
                 num_layers: int = 0):
        """Initializer.

        Args:
            fiber: Fiber() of feature multiplicities and types
            nonlin: nonlinearity to use everywhere
            num_layers: non-negative number of linear layers in fnc
        """
        super().__init__()
        self.fiber = fiber
        self.nonlin = nonlin
        self.num_layers = num_layers

        # Regularization for computing phase: gradients explode otherwise
        self.eps = 1e-12

        # Norm mappings: 1 per feature type
        self.bias = nn.ParameterDict()
        for m, d in self.fiber.structure:
            self.bias[str(d)] = nn.Parameter(torch.randn(m).view(1, m))

    def __repr__(self):
        return f"GNormTFN()"


    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            # Compute the norms and normalized features
            # v shape: [...,m , 2*k+1]
            norm = v.norm(2, -1, keepdim=True).clamp_min(self.eps).expand_as(v)
            phase = v / norm

            # Transform on norms
            # transformed = self.transform[str(k)](norm[..., 0]).unsqueeze(-1)
            transformed = self.nonlin(norm[..., 0] + self.bias[str(k)])

            # Nonlinearity on norm
            output[k] = (transformed.unsqueeze(-1) * phase).view(*v.shape)

        return output


class GAttentiveSelfInt(nn.Module):

    def __init__(self, f_in, f_out):
        """SE(3)-equivariant 1x1 convolution.

        Args:
            f_in: input Fiber() of feature multiplicities and types
            f_out: output Fiber() of feature multiplicities and types
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.nonlin = nn.LeakyReLU()
        self.num_layers = 2
        self.eps = 1e-12 # regularisation for phase: gradients explode otherwise

        # one network for attention weights per degree
        self.transform = nn.ModuleDict()
        for o, m_in in self.f_in.structure_dict.items():
            m_out = self.f_out.structure_dict[o]
            self.transform[str(o)] = self._build_net(m_in, m_out)

    def __repr__(self):
        return f"AttentiveSelfInteractionSE3(in={self.f_in}, out={self.f_out})"

    def _build_net(self, m_in: int, m_out):
        n_hidden = m_in * m_out
        cur_inpt = m_in * m_in
        net = []
        for i in range(1, self.num_layers):
            net.append(nn.LayerNorm(int(cur_inpt)))
            net.append(self.nonlin)
            # TODO: implement cleaner init
            net.append(
                nn.Linear(cur_inpt, n_hidden, bias=(i == self.num_layers - 1)))
            nn.init.kaiming_uniform_(net[-1].weight)
            cur_inpt = n_hidden
        return nn.Sequential(*net)

    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            # v shape: [..., m, 2*k+1]
            first_dims = v.shape[:-2]
            m_in  = self.f_in.structure_dict[int(k)]
            m_out = self.f_out.structure_dict[int(k)]
            assert v.shape[-2] == m_in
            assert v.shape[-1] == 2 * int(k) + 1

            # Compute the norms and normalized features
            #norm = v.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps).expand_as(v)
            #phase = v / norm # [..., m, 2*k+1]
            scalars = torch.einsum('...ac,...bc->...ab', [v, v]) # [..., m_in, m_in]
            scalars = scalars.view(*first_dims, m_in*m_in) # [..., m_in*m_in]
            sign = scalars.sign()
            scalars = scalars.abs_().clamp_min(self.eps)
            scalars *= sign

            # perform attention
            att_weights = self.transform[str(k)](scalars) # [..., m_out*m_in]
            att_weights = att_weights.view(*first_dims, m_out, m_in) # [..., m_out, m_in]
            att_weights = F.softmax(input=att_weights, dim=-1)
            # shape [..., m_out, 2*k+1]
            # output[k] = torch.einsum('...nm,...md->...nd', [att_weights, phase])
            output[k] = torch.einsum('...nm,...md->...nd', [att_weights, v])

        return output



class GNormSE3(nn.Module):
    """Graph Norm-based SE(3)-equivariant nonlinearity.

    Nonlinearities are important in SE(3) equivariant GCNs. They are also quite
    expensive to compute, so it is convenient for them to share resources with
    other layers, such as normalization. The general workflow is as follows:

    > for feature type in features:
    >    norm, phase <- feature
    >    output = fnc(norm) * phase

    where fnc: {R+}^m -> R^m is a learnable map from m norms to m scalars.
    """
    def __init__(self, fiber, nonlin=nn.ReLU(inplace=True), num_layers: int=0):
        """Initializer.

        Args:
            fiber: Fiber() of feature multiplicities and types
            nonlin: nonlinearity to use everywhere
            num_layers: non-negative number of linear layers in fnc
        """
        super().__init__()
        self.fiber = fiber
        self.nonlin = nonlin
        self.num_layers = num_layers

        # Regularization for computing phase: gradients explode otherwise
        self.eps = 1e-12

        # Norm mappings: 1 per feature type
        self.transform = nn.ModuleDict()
        for m, d in self.fiber.structure:
            self.transform[str(d)] = self._build_net(int(m))

    def __repr__(self):
         return f"GNormSE3(num_layers={self.num_layers}, nonlin={self.nonlin})"

    def _build_net(self, m: int):
        net = []
        for i in range(self.num_layers):
            net.append(BN(int(m)))
            net.append(self.nonlin)
            # TODO: implement cleaner init
            net.append(nn.Linear(m, m, bias=(i==self.num_layers-1)))
            nn.init.kaiming_uniform_(net[-1].weight)
        if self.num_layers == 0:
            net.append(BN(int(m)))
            net.append(self.nonlin)
        return nn.Sequential(*net)

    @profile
    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            # Compute the norms and normalized features
            # v shape: [...,m , 2*k+1]
            norm = v.norm(2, -1, keepdim=True).clamp_min(self.eps).expand_as(v)
            phase = v / norm

            # Transform on norms
            transformed = self.transform[str(k)](norm[...,0]).unsqueeze(-1)

            # Nonlinearity on norm
            output[k] = (transformed * phase).view(*v.shape)

        return output


class BN(nn.Module):
    """SE(3)-equvariant batch/layer normalization"""
    def __init__(self, m):
        """SE(3)-equvariant batch/layer normalization

        Args:
            m: int for number of output channels
        """
        super().__init__()
        self.bn = nn.LayerNorm(m)

    def forward(self, x):
        return self.bn(x)


class GConvSE3Partial(nn.Module):
    """Graph SE(3)-equivariant node -> edge layer"""
    def __init__(self, f_in, f_out, edge_dim: int=0, x_ij=None):
        """SE(3)-equivariant partial convolution.

        A partial convolution computes the inner product between a kernel and
        each input channel, without summing over the result from each input
        channel. This unfolded structure makes it amenable to be used for
        computing the value-embeddings of the attention mechanism.

        Args:
            f_in: list of tuples [(multiplicities, type),...]
            f_out: list of tuples [(multiplicities, type),...]
        """
        super().__init__()
        self.f_out = f_out
        self.edge_dim = edge_dim

        # adding/concatinating relative position to feature vectors
        # 'cat' concatenates relative position & existing feature vector
        # 'add' adds it, but only if multiplicity > 1
        assert x_ij in [None, 'cat', 'add']
        self.x_ij = x_ij
        if x_ij == 'cat':
            self.f_in = Fiber.combine(f_in, Fiber(structure=[(1,1)]))
        else:
            self.f_in = f_in

        # Node -> edge weights
        self.kernel_unary = nn.ModuleDict()
        for (mi, di) in self.f_in.structure:
            for (mo, do) in self.f_out.structure:
                self.kernel_unary[f'({di},{do})'] = PairwiseConv(di, mi, do, mo, edge_dim=edge_dim)

    def __repr__(self):
        return f'GConvSE3Partial(structure={self.f_out})'

    def udf_u_mul_e(self, d_out,batch_size,num_nodes,num_edges):
        """Compute the partial convolution for a single output feature type.

        This function is set up as a User Defined Function in DGL.

        Args:
            d_out: output feature type
        Returns:
            node -> edge function handle
        """
        def fnc(edges):
            # Neighbor -> center messages
            msg = 0

            for m_in, d_in in self.f_in.structure:

                # if type 1 and flag set, add relative position as feature
                if self.x_ij == 'cat' and d_in == 1:
                    # relative positions
                    rel = (edges.dst['x'] - edges.src['x']).view(-1, 3, 1)
                    rel = (edges.dst['x'] - edges.src['x']).view(-1, 3, 1)
                    m_ori = m_in - 1
                    if m_ori == 0:
                        # no type 1 input feature, just use relative position
                        src = rel
                    else:
                        # features of src node, shape [edges, m_in*(2l+1), 1]
                        src = edges.src[f'{d_in}'].view(-1, m_ori*(2*d_in+1), 1)
                        # add to feature vector
                        src = torch.cat([src, rel], dim=1)
                elif self.x_ij == 'add' and d_in == 1 and m_in > 1:
                    src = edges.src[f'{d_in}'].view(-1, m_in*(2*d_in+1), 1)
                    rel = (edges.dst['x'] - edges.src['x']).view(-1, 3, 1)
                    rel = edges.src['x'].view(-1, 3, 1)
                    src[..., :3, :1] = src[..., :3, :1] + rel
                else:
                    src = edges.src[f'{d_in}'].view(-1, m_in*(2*d_in+1), 1)#velocity

                    '''
                    dim1,dim2 = src.shape[-1],src.shape[-2]

                    if dim2 == 3:
                        print(edges.dst['x'].view(-1,,19,3,1)[:,:,0,:,:])## other points
                        
                        print(edges.src['x'].view(-1,20,19,3,1)[:,:,0,:,:])#x_0
                        print(src) #v_0
                        #print(src.view(-1,20,19,dim2,dim1)[:,:,0,:,:].squeeze(-3).view(-1,dim2,dim1).shape)
                        #print(src.shape)
                    '''
                edge = edges.data[f'({d_in},{d_out})'] ## This is W, k-->l (indeed the concatenation makes dim double)

                ## reduce computation to linear
                e1,e2 = edge.shape[-1],edge.shape[-2]
                s1,s2 = src.shape[-1],src.shape[-2]
                edge = edge.view(batch_size,num_nodes,num_edges,e2,e1)

                edge = edge[:,:,0,:,:]
                edge = edge.view(batch_size*num_nodes,e2,e1)

                #print(f'src shape is {src.shape}')
                src = src.view(batch_size,num_nodes,num_edges,s2,s1)
                src = src[:,:,0,:,:]
                src = src.view(batch_size*num_nodes,s2,s1)

                #print(edges.data)
                '''
                print(edge.shape)
                print(src.shape)
                print(torch.matmul(edge, src).shape)
                for i in range(380):
                    print(edge[i,:,:])
                    
                ## confirm identical across edge
                '''

                msg = msg + torch.matmul(edge, src)
            m1,m2 = msg.shape[-1],msg.shape[-2]
            msg = msg.view(batch_size,num_nodes,m2,m1,1)
            msg = msg.expand(batch_size,num_nodes,m2,m1,num_edges)
            msg = msg.permute(0,1,-1,2,3).contiguous()
            msg = msg.view(batch_size*num_nodes*num_edges,m2,m1)
            #print(f'msg shape is {msg.shape}')
            msg = msg.view(msg.shape[0], -1, 2*d_out+1)

            return {f'out{d_out}': msg.view(msg.shape[0], -1, 2*d_out+1)}
        return fnc

    @profile
    def forward(self, h, G=None, r=None, basis=None, **kwargs):
        """Forward pass of the linear layer

        Args:
            h: dict of node-features
            G: minibatch of (homo)graphs
            r: inter-atomic distances
            basis: pre-computed Q * Y
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """
        batch_size = G.batch_size
        num_nodes = G.num_nodes()//batch_size
        num_edges = G.num_edges()//(batch_size*num_nodes)

        with G.local_scope():
            # Add node features to local graph scope
            for k, v in h.items():
                #print(f'key of dic is {k}')
                #print(f'value shape of dic is {v.shape}')
                G.ndata[k] = v

            # Add edge features
            if 'w' in G.edata.keys():
                w = G.edata['w'] # shape: [#edges_in_batch, #bond_types]
                feat = torch.cat([w, r], -1)

                #feat = (w*r).expand(-1,2)
            else:
                feat = torch.cat([r, ], -1)

            feat = feat.view(batch_size,num_nodes,num_edges,-1)
            feat = feat[:,:,0,:]
            feat = feat.view(batch_size*num_nodes,-1)

            for (mi, di) in self.f_in.structure:
                for (mo, do) in self.f_out.structure:
                    etype = f'({di},{do})'



                    unary = self.kernel_unary[etype](feat, basis)
                    dim1,dim2 = unary.shape[-1],unary.shape[-2]

                    unary = unary.view(batch_size,num_nodes,dim2,dim1,1)
                    unary = unary.expand(batch_size,num_nodes,dim2,dim1,num_edges)
                    unary = unary.permute(0,1,-1,2,3).contiguous()# reshape to BxNxExFx3
                    unary = unary.view(batch_size*num_nodes*num_edges,dim2,dim1)## flatten

                    #print(f'unary shape is {self.kernel_unary[etype](feat, basis).shape}')
                    G.edata[etype] = unary #self.kernel_unary[etype](feat, basis)

            # Perform message-passing for each output feature type
            for d in self.f_out.degrees:
                G.apply_edges(self.udf_u_mul_e(d,batch_size,num_nodes,num_edges))

            return {f'{d}': G.edata[f'out{d}'] for d in self.f_out.degrees}


class GMABSE3(nn.Module):
    """An SE(3)-equivariant multi-headed self-attention module for DGL graphs."""
    def __init__(self, f_value: Fiber, f_key: Fiber, n_heads: int,Performer,max_rf,antithetic):
        """SE(3)-equivariant MAB (multi-headed attention block) layer.

        Args:
            f_value: Fiber() object for value-embeddings
            f_key: Fiber() object for key-embeddings
            n_heads: number of heads
        """
        super().__init__()
        self.f_value = f_value
        self.f_key = f_key
        self.n_heads = n_heads
        self.new_dgl = version.parse(dgl.__version__) > version.parse('0.4.4')
        self.Performer = Performer
        self.max_rf = max_rf
        self.antithetic = antithetic

    def __repr__(self):
        return f'GMABSE3(n_heads={self.n_heads}, structure={self.f_value})'



    def udf_u_mul_e(self, d_out):
        """Compute the weighted sum for a single output feature type.

        This function is set up as a User Defined Function in DGL.

        Args:
            d_out: output feature type
        Returns:
            edge -> node function handle
        """
        def fnc(edges):
            # Neighbor -> center messages
            attn = edges.data['a']
            value = edges.data[f'v{d_out}']

            # Apply attention weights
            msg = attn.unsqueeze(-1).unsqueeze(-1) * value

            return {'m': msg}
        return fnc
        


    @profile

    def Phi(self,z,w):
        '''

        z: Q or K with shape B H L d
        w: random features with shape m d
        return: random feature approximation
        '''
        m = w.shape[0]
        scaler = torch.exp(-(z*z).sum(-1)/2)/np.sqrt(m) ## B H L
        z = z.unsqueeze(-1).contiguous().expand(z.shape[0],z.shape[1],z.shape[2],z.shape[3],m)## B H L d m
        z = z.transpose(-1,-2).contiguous() ## B H L m d
        z_prime = (z*w).sum(-1) ## dot product of z and w, shape B H L m
        z_prime = z_prime.permute(-1,0,1,2).contiguous() ## m B H L

        return (z_prime*scaler).permute(1,2,3,0) ##B H L m
    def forward(self, v, k: Dict=None, q: Dict=None, G=None, **kwargs):
        """Forward pass of the linear layer

        Args:
            G: minibatch of (homo)graphs
            v: dict of value edge-features
            k: dict of key edge-features
            q: dict of query node-features
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """
        with G.local_scope():
            # Add node features to local graph scope
            ## We use the stacked tensor representation for attention
            batch_size = G.batch_size
            L = G.num_nodes()//batch_size
            V = {}
            #print(self.f_value.structure)
            for m, d in self.f_value.structure:
                #G.edata[f'v{d}'] = v[f'{d}'].view(-1, self.n_heads, m//self.n_heads, 2*d+1)
                V[f'v{d}'] = v[f'{d}'].view(-1, self.n_heads, m // self.n_heads, 2 * d + 1)
            #G.edata['k'] = fiber2head(k, self.n_heads, self.f_key, squeeze=True) # [edges, heads, channels](?)
            #G.ndata['q'] = fiber2head(q, self.n_heads, self.f_key, squeeze=True) # [nodes, heads, channels](?)



            Q = fiber2head(k, self.n_heads, self.f_key, squeeze=True) # [edges, heads, channels](?)
            K = fiber2head(q, self.n_heads, self.f_key, squeeze=True) # [nodes, heads, channels](?)


            q1,q2 = Q.shape[-1],Q.shape[-2]
            k1, k2 = K.shape[-1], K.shape[-2]
            d_scaler = np.sqrt(self.f_key.n_features)
            Q = Q.view(batch_size,L,q2,q1)## BxLxHxd
            K = Q.view(batch_size, L, k2, k1)
            Q = Q.permute(0,2,1,3).contiguous()## BxHxLxd
            K = K.permute(0, 2, 1, 3).contiguous()

            #print(f'q shape is {Q.shape}')
            #print(f'k shape is {K.shape}')


            if self.Performer:
                #This is the performer setting

                if self.antithetic:
                    w = gaussian_orthogonal_random_matrix(nb_rows=self.max_rf//2,nb_columns=q1)
                    w = torch.cat([w,-w],dim = 0).cuda() ## 8*d, antithetic
                else:
                    w = gaussian_orthogonal_random_matrix(nb_rows=self.max_rf, nb_columns=q1)

                num_rand_feat = w.shape[0]

                #K_prime = self.Phi(K/np.sqrt(d_scaler),w) #.transpose(-1,-2).contiguous()  # B H m L
                #Q_prime = self.Phi(Q / np.sqrt(d_scaler),w) ## B H L m
                K_prime = softmax_kernel(data = K,projection_matrix = w,is_query = False)
                Q_prime = softmax_kernel(data=  Q, projection_matrix=w, is_query=True)
                #print(self.f_key.n_features,q1)
                output = {}
                for d in self.f_value.degrees:
                    v = V[f'v{d}']
                    _,head,chan,d_out = v.shape

                    v = v.view(batch_size,L,head,chan,d_out)##BxLxHxCxd_out
                    v = v.permute(0,2,3,1,-1).contiguous() ## BxHxCxLxd_out
                    #k = K_prime.unsqueeze(-1).expand(batch_size,self.n_heads,num_rand_feat,L,chan).permute(0,1,-1,2,3).contiguous()        ## B H m L C --> B H C m L then KV = B H C m d_out
                    q = Q_prime.unsqueeze(-1).expand(batch_size,self.n_heads,L,num_rand_feat, chan).permute(0, 1, -1, 2, 3).contiguous() ## B H L m C --> B H C L m QKV = B H C L d_out
                    k = K_prime.unsqueeze(-1).expand(batch_size, self.n_heads,L, num_rand_feat, chan).permute(0, 1, -1,2,3).contiguous()
                    #v = torch.matmul(q,torch.matmul(k,v))  # B H C L d_out
                    v = linear_attention(q,k,v)
                    v = v.view(batch_size,self.n_heads*chan,L,d_out)## B H*C L d_out
                    v = v.permute(0,2,1,-1).contiguous()# B L C dout
                    v = v.view(batch_size*L,chan*self.n_heads,d_out)
                    output[f'{d}'] = v
                return output



            ##

            A = torch.matmul(Q,K.transpose(-1,-2).contiguous())/d_scaler## should be BxHxLxL

            '''masking
            mask = torch.zeros(size = (L,L)).cuda()

            mask = mask.fill_diagonal_(-torch.inf).cuda()
            A = A + mask ##
            del mask
            '''
            A = A.softmax(-1)


            #print(f'attn shape is{A.shape}')
            
            output = {}
            for d in self.f_value.degrees:
                v = V[f'v{d}']
                _,head,chan,d_out = v.shape
                attn = A.view(batch_size,self.n_heads,L,L,1)
                attn = attn.expand(batch_size,self.n_heads,L,L,chan)## B H L L C
                #print(f'expanded attention shape{attn.shape}')
                attn = attn.permute(0,1,-1,2,3).contiguous() # B H C L L
                v = v.view(batch_size,L,head,chan,d_out)##BxLxHxCxd_out
                v = v.permute(0,2,3,1,-1).contiguous() ## BxHxCxLxd_out

                v = torch.matmul(attn,v) ## B H C L d_out

                v = v.view(batch_size,self.n_heads*chan,L,d_out)## B H*C L d_out
                v = v.permute(0, 2, 1, -1).contiguous()  # B L C dout

                #print(v.shape)
                v = v.view(batch_size*L,chan*self.n_heads,d_out)
                output[f'{d}'] = v
                #print(v.shape)

            return output
            



'''

                ## output is BxLxcxd_out sum over heads
                #v1,v2 = v.shape[-1],v.shape[-2]
                #v = v.view(batch_size,L,v2,v1)
                print(f'value shape is {v.shape}')
                #output[f'{d}'] = torch.matmul(A,V[f'v{d}'])
                



            G.apply_edges(fn.e_dot_v('k', 'q', 'e'))

            ## Apply softmax
            e = G.edata.pop('e')


            if self.new_dgl:
                # in dgl 5.3, e has an extra dimension compared to dgl 4.3
                # the following, we get rid of this be reshaping
                n_edges = G.edata['k'].shape[0]
                e = e.view([n_edges, self.n_heads])
            e = e / np.sqrt(self.f_key.n_features)
            G.edata['a'] = edge_softmax(G, e)
            # Perform attention-weighted message-passing


            for d in self.f_value.degrees:
                G.update_all(self.udf_u_mul_e(d), fn.sum('m', f'out{d}'))

            output = {}
            for m, d in self.f_value.structure:
                output[f'{d}'] = G.ndata[f'out{d}'].view(-1, m, 2*d+1)
                print(m)
                print(f" output shape {output[f'{d}'].shape}")

            return output
'''

class GSE3Res(nn.Module):
    """Graph attention block with SE(3)-equivariance and skip connection"""
    def __init__(self, f_in: Fiber, f_out: Fiber, edge_dim: int=0, div: float=4,
                 n_heads: int=1, learnable_skip=True, skip='cat', selfint='1x1', x_ij=None,Performer = False,max_rf = 8,antithetic = True):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.div = div
        self.n_heads = n_heads
        self.skip = skip  # valid: 'cat', 'sum', None
        self.max_rf = max_rf
        self.antithetic = antithetic

        # f_mid_out has same structure as 'f_out' but #channels divided by 'div'
        # this will be used for the values
        f_mid_out = {k: int(v // div) for k, v in self.f_out.structure_dict.items()}
        self.f_mid_out = Fiber(dictionary=f_mid_out)

        # f_mid_in has same structure as f_mid_out, but only degrees which are in f_in
        # this will be used for keys and queries
        # (queries are merely projected, hence degrees have to match input)
        f_mid_in = {d: m for d, m in f_mid_out.items() if d in self.f_in.degrees}
        self.f_mid_in = Fiber(dictionary=f_mid_in)

        self.edge_dim = edge_dim

        self.GMAB = nn.ModuleDict()

        # Projections
        self.GMAB['v'] = GConvSE3Partial(f_in, self.f_mid_out, edge_dim=edge_dim, x_ij=x_ij)
        self.GMAB['k'] = GConvSE3Partial(f_in, self.f_mid_in, edge_dim=edge_dim, x_ij=x_ij)
        #self.GMAB['q'] = G1x1SE3(f_in, self.f_mid_in)
        self.GMAB['q'] = GConvSE3Partial(f_in, self.f_mid_in, edge_dim=edge_dim, x_ij=x_ij)

        # Attention
        self.Performer = Performer
        self.GMAB['attn'] = GMABSE3(self.f_mid_out, self.f_mid_in, n_heads=n_heads,Performer=self.Performer,max_rf=self.max_rf,antithetic=self.antithetic)

        # Skip connections
        if self.skip == 'cat':
            self.cat = GCat(self.f_mid_out, f_in)
            if selfint == 'att':
                self.project = GAttentiveSelfInt(self.cat.f_out, f_out)
            elif selfint == '1x1':
                self.project = G1x1SE3(self.cat.f_out, f_out, learnable=learnable_skip)
        elif self.skip == 'sum':
            self.project = G1x1SE3(self.f_mid_out, f_out, learnable=learnable_skip)
            self.add = GSum(f_out, f_in)
            # the following checks whether the skip connection would change
            # the output fibre strucure; the reason can be that the input has
            # more channels than the ouput (for at least one degree); this would
            # then cause a (hard to debug) error in the next layer
            assert self.add.f_out.structure_dict == f_out.structure_dict, \
                'skip connection would change output structure'

    @profile
    def forward(self, features, G, **kwargs):
        # Embeddings

        v = self.GMAB['v'](features, G=G, **kwargs)
        k = self.GMAB['k'](features, G=G, **kwargs)
        q = self.GMAB['q'](features, G=G,**kwargs)
        batch_size = G.batch_size
        num_nodes = G.num_nodes() // batch_size
        num_edges = G.num_edges() // (batch_size*num_nodes)

        for key in q.keys():
            #print(q[key].shape)
            dim1,dim2 = q[key].shape[-1],q[key].shape[-2]
            q[key] = q[key].view(-1,num_nodes,num_edges,dim2,dim1)[:,:,0,:,:].view(-1,dim2,dim1)

        for key in k.keys():
            #print(k[key].shape)
            dim1,dim2 = k[key].shape[-1],k[key].shape[-2]
            k[key] = k[key].view(-1,num_nodes,num_edges,dim2,dim1)[:,:,0,:,:].view(-1,dim2,dim1)

        for key in v.keys():
            #print(v[key].shape)
            dim1,dim2 = v[key].shape[-1],v[key].shape[-2]
            v[key] = v[key].view(-1,num_nodes,num_edges,dim2,dim1)[:,:,0,:,:].view(-1,dim2,dim1)

        # Attention
        z = self.GMAB['attn'](v, k=k, q=q, G=G)


        if self.skip == 'cat':
            z = self.cat(z, features)
            z = self.project(z)
        elif self.skip == 'sum':
            # Skip + residual
            z = self.project(z)
            z = self.add(z, features)
        return z

### Helper and wrapper functions

class GSum(nn.Module):
    """SE(3)-equvariant graph residual sum function."""
    def __init__(self, f_x: Fiber, f_y: Fiber):
        """SE(3)-equvariant graph residual sum function.

        Args:
            f_x: Fiber() object for fiber of summands
            f_y: Fiber() object for fiber of summands
        """
        super().__init__()
        self.f_x = f_x
        self.f_y = f_y
        self.f_out = Fiber.combine_max(f_x, f_y)

    def __repr__(self):
        return f"GSum(structure={self.f_out})"

    def forward(self, x, y):
        out = {}
        for k in self.f_out.degrees:
            k = str(k)
            if (k in x) and (k in y):
                if x[k].shape[1] > y[k].shape[1]:
                    diff = x[k].shape[1] - y[k].shape[1]
                    zeros = torch.zeros(x[k].shape[0], diff, x[k].shape[2]).to(y[k].device)
                    y[k] = torch.cat([y[k], zeros], 1)
                elif x[k].shape[1] < y[k].shape[1]:
                    diff = y[k].shape[1] - x[k].shape[1]
                    zeros = torch.zeros(x[k].shape[0], diff, x[k].shape[2]).to(y[k].device)
                    x[k] = torch.cat([x[k], zeros], 1)

                out[k] = x[k] + y[k]
            elif k in x:
                out[k] = x[k]
            elif k in y:
                out[k] = y[k]
        return out


class GCat(nn.Module):
    """Concat only degrees which are in f_x"""
    def __init__(self, f_x: Fiber, f_y: Fiber):
        super().__init__()
        self.f_x = f_x
        self.f_y = f_y
        f_out = {}
        for k in f_x.degrees:
            f_out[k] = f_x.dict[k]
            if k in f_y.degrees:
                f_out[k] += f_y.dict[k]
        self.f_out = Fiber(dictionary=f_out)

    def __repr__(self):
        return f"GCat(structure={self.f_out})"

    def forward(self, x, y):
        out = {}
        for k in self.f_out.degrees:
            k = str(k)
            if k in y:
                out[k] = torch.cat([x[k], y[k]], 1)
            else:
                out[k] = x[k]
        return out


class GAvgPooling(nn.Module):
    """Graph Average Pooling module."""
    def __init__(self, type='0'):
        super().__init__()
        self.pool = AvgPooling()
        self.type = type

    @profile
    def forward(self, features, G, **kwargs):
        if self.type == '0':
            h = features['0'][...,-1]
            pooled = self.pool(G, h)
        elif self.type == '1':
            pooled = []
            for i in range(3):
                h_i = features['1'][..., i]
                pooled.append(self.pool(G, h_i).unsqueeze(-1))
            pooled = torch.cat(pooled, axis=-1)
            pooled = {'1': pooled}
        else:
            print('GAvgPooling for type > 0 not implemented')
            exit()
        return pooled


class GMaxPooling(nn.Module):
    """Graph Max Pooling module."""
    def __init__(self):
        super().__init__()
        self.pool = MaxPooling()

    @profile
    def forward(self, features, G, **kwargs):
        h = features['0'][...,-1]
        return self.pool(G, h)


