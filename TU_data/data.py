
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import re

r"""
        This is a simplified version of torch_geometric.data.Data
        We do not use nodes/edges attributes that some datasets have.
"""

class Data(object):
    r"""
    Args:
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
    """

    def __init__(self, edge_index=None, y=None):
        self.edge_index = edge_index
        self.y = y

    @classmethod
    def from_dict(cls, dictionary):
        data = cls()

        for key, item in dictionary.items():
            data[key] = item

        if torch_geometric.is_debug_enabled():
            data.debug()

        return data

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @property
    def keys(self):
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __len__(self):
        return len(self.keys)

    def __contains__(self, key):
        return key in self.keys

    def __iter__(self):
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    def __inc__(self, key, value):
        return self.num_nodes if bool(re.search('(index|face)', key)) else 0

    @property
    def num_nodes(self):
        if hasattr(self, '__num_nodes__'):
            return self.__num_nodes__
        if self.edge_index is not None:
            warnings.warn(__num_nodes_warn_msg__.format('edge'))
            return maybe_num_nodes(self.edge_index)
        return None

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    @property
    def num_edges(self):
        for key, item in self('edge_index', 'edge_attr'):
            return item.size(self.__cat_dim__(key, item))
        return None

    
    def clone(self):
        return self.__class__.from_dict({
            k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
            for k, v in self.__dict__.items()
        })
