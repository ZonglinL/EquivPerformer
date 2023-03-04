import torch
from torch import nn
from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias, GConvSE3, GMaxPooling, GAvgPooling,GNormSE3, AttentionPooling
from equivariant_attention.fibers import Fiber


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""

    def __init__(self, num_layers: int, num_channels: int, num_degrees: int = 4, div: float = 4,
                 n_heads: int = 1, si_m='1x1', si_e='att', x_ij='add', kernel=True, num_random=5,
                 out_dim=128*2, num_class=15, batch=16, antithetic=False, num_points=128):
        """
        Args:
            num_layers: number of attention layers
            num_channels: number of channels per degree
            num_degrees: number of degrees (aka types) in hidden layer, count start from type-0
            div: (int >= 1) keys, queries and values will have (num_channels/div) channels
            n_heads: (int >= 1) for multi-headed attention
            si_m: ['1x1', 'att'] type of self-interaction in hidden layers
            si_e: ['1x1', 'att'] type of self-interaction in final layer
            x_ij: ['add', 'cat'] use relative position as edge feature
            kernel: bool whether to use performer
            nb_features: int number of random features
            batch: batch size
        """
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = 0
        self.div = div
        self.n_heads = n_heads
        self.si_m, self.si_e = si_m, si_e
        self.x_ij = x_ij
        #self.out_dim = out_dim
        self.num_class = num_class
        self.batch = batch
        self.num_points = num_points
        self.out_dim = 64

        self.fibers = {'in': Fiber(dictionary={1:1}),
                       'mid': Fiber(self.num_degrees, self.num_channels),
                       'out': Fiber(dictionary={0: self.out_dim})}

        self.region_fibers = {'in': Fiber(dictionary={1:self.num_channels}),
                       'mid': Fiber(self.num_degrees, self.num_channels),
                       'out': Fiber(dictionary={0: self.out_dim})}

        self.sub_fibers = {'in': Fiber(dictionary={1: 1}),
                              'mid': Fiber(self.num_degrees, self.num_channels),
                              'out': Fiber(dictionary={1: self.num_channels})}

        self.kernel = kernel
        self.num_random = num_random
        self.antithetic = antithetic

        self.Gblock = self._build_gcn(self.fibers)
        self.sub_Gblock = self._build_gcn(self.sub_fibers)
        self.region_Gblock = self._build_gcn(self.region_fibers)

        #self.pooling = GAvgPooling()

        self.pooling = AttentionPooling(self.out_dim)
        self.sub_pooling = AttentionPooling(self.out_dim)
        self.whole_pooling = AttentionPooling(self.out_dim)
        self.decoder = nn.Sequential(nn.Linear(self.out_dim,self.out_dim), nn.Linear(self.out_dim,self.num_class))
        '''
        self.hid = torch.nn.Linear(self.out_dim, 64)
        self.act = torch.nn.Identity()
        self.out = torch.nn.Linear(64, self.num_class)
        '''

        print(self.Gblock)

    def _build_gcn(self, fibers):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']

        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, div=self.div, n_heads=self.n_heads,
                                  learnable_skip=True, skip='cat', selfint=self.si_m, x_ij=self.x_ij, kernel=self.kernel,
                                  num_random=self.num_random, antithetic=self.antithetic))
            Gblock.append(GNormBias(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(
            GSE3Res(fibers['mid'], fibers['out'], edge_dim=self.edge_dim, div=1, n_heads=self.n_heads,
                    learnable_skip=True, skip='cat', selfint=self.si_e, x_ij=self.x_ij, kernel=self.kernel,
                    num_random=self.num_random, antithetic=self.antithetic))
        return nn.ModuleList(Gblock)

    def forward(self, G,sub_G,region_G):
        # Compute equivariant weight basis from relative positions
        global_basis, global_r = get_basis_and_r(G, self.num_degrees-1)

        global_enc = {'1': torch.zeros_like(G.ndata['x'])}

        #h_enc = {'1':G.ndata['x']}
        for layer in self.Gblock:
            global_enc = layer(global_enc, G=G, r=global_r, basis=global_basis)
        # B*N, 2, 3 ==> B, 3, 2*N
        # enc = h_enc['1'].view(self.batch, 3, -1)
        #h_enc = self.pooling(h_enc,G)
        global_enc = global_enc['0'].view(self.batch, -1, self.out_dim)
        global_enc = self.pooling(global_enc).view(self.batch,-1,self.out_dim) ## batch dim
        #h_enc = global_enc

        local_basis, local_r = get_basis_and_r(sub_G, self.num_degrees-1)

        local_enc = {'1': torch.zeros_like(sub_G.ndata['x'])}
        
        for layer in self.sub_Gblock:
            local_enc = layer(local_enc, G=sub_G, r=local_r, basis=local_basis)

        ##print(local_enc['1'].shape) ## ... chan degree
        local_enc = local_enc['1'].view(-1,16,self.num_channels,3)## batch*num_corners num_neighbors channel 3
        local_enc = local_enc.mean(1) ## batch*num_corners channel 3
        '''
        local_enc = local_enc['0'].view(-1,16,self.out_dim) ## batch*num_graphs num_neighbors dim
        local_enc = self.sub_pooling(local_enc) ## batch*num_graphs dim
        local_enc = torch.stack(torch.split(local_enc,self.batch),1) ## batch num_graphs dim
        '''
        #region_enc = torch.stack(torch.split(local_enc,self.batch),1) ## batch num_corners channel 3
        region_enc = local_enc
        #print(region_enc.shape)
        region_basis, region_r = get_basis_and_r(region_G, self.num_degrees - 1)

        region_enc = {'1': region_enc}
        for layer in self.region_Gblock:
            region_enc = layer(region_enc, G=region_G, r=region_r , basis=region_basis)

        region_enc = region_enc['0'].view((self.batch, -1, self.out_dim)) ## batch num_corners out_dim

        h_enc = torch.cat([global_enc, region_enc], 1)  ## batch numgraphs+1 dim


        #h_enc = torch.cat([global_enc,local_enc],1) ## batch numgraphs+1 dim
        h_enc = self.whole_pooling(h_enc) ## batch dim
        
        probs = self.decoder(h_enc.view(self.batch,-1))

        return probs


class TFN(nn.Module):
    """Tensorfiel Network"""

    def __init__(self, num_layers: int, num_channels: int, num_degrees: int = 4, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = 1

        self.fibers = {'in': Fiber(dictionary={0: 1, 1: 1}),
                       'mid': Fiber(self.num_degrees, self.num_channels),
                       'out': Fiber(dictionary={1: 2})}

        blocks = self._build_gcn(self.fibers)
        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)
        # purely for counting paramters in utils_logging.py
        self.enc, self.dec = self.Gblock, self.FCblock

    def _build_gcn(self, fibers):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']

        for i in range(self.num_layers-1):
            Gblock.append(GConvSE3(fin, fibers['mid'], self_interaction=True, flavor='TFN', edge_dim=self.edge_dim))
            Gblock.append(GNormBias(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(
            GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, flavor='TFN', edge_dim=self.edge_dim))

        return nn.ModuleList(Gblock), nn.ModuleList([])

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h_enc = {'0': G.ndata['c'], '1': G.ndata['v']}
        for layer in self.Gblock:
            h_enc = layer(h_enc, G=G, r=r, basis=basis)

        return h_enc['1']
