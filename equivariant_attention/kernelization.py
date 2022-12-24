

import math
import torch

from einops import rearrange, repeat


from distutils.version import LooseVersion

TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')

def orthogonal_matrix_chunk(cols, device=None):
    # normal matrix
    unstructured_block = torch.randn((cols, cols), device=device)
    # (10, 10)
    # exactly the same thing; just in different version
    if TORCH_GE_1_8_0:
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    else:
        q, r = torch.qr(unstructured_block.cpu(), some=True)

    q, r = map(lambda t: t.to(device), (q, r))
    # (10, 10)
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    # e.g. 10, 10 ==> nb_full_blocks=1
    nb_full_blocks = int(nb_rows / nb_columns)
    #
    block_list = []
    # 1 repeat
    for _ in range(nb_full_blocks):
        # 10,10
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        # (10, 10)
        block_list.append(q)
    # tall case is special: do this for tall case(i.e. nb_columns< nb_rows)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns

    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])
    # same dimension as (nb_rows, nb_columns)
    final_matrix = torch.cat(block_list)

    # normalization(??)
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):

    b, h, *_ = data.shape
    #
    data_normalizer = ((data.shape[1]*data.shape[-1]) ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)
    a = data_normalizer * data
    #print(a.shape, projection.shape)
    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                      torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                      torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)

    return data_dash.type_as(data)


def linear_attention(q, k, v):
 k_cumsum = k.sum(dim = -2)
 D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
 context = torch.einsum('...nd,...ne->...de', k, v)
 out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
 return out