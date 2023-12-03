from collections import namedtuple
from visualize import plot


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Arch = {5: Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6)), 6: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 3), ('sep_conv_3x3', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('skip_connect', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)), 4: Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)), 7: Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 3), ('sep_conv_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6)), 2: Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 1), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 0), ('skip_connect', 1)], reduce_concat=range(2, 6)), 1: Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 3), ('sep_conv_3x3', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)), 0: Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 3), ('sep_conv_3x3', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3', 3), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6)), 3: Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 3), ('sep_conv_3x3', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('sep_conv_3x3', 3), ('skip_connect', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))}
# Arch = {7: Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6)), 3: Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('skip_connect', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('skip_connect', 1), ('skip_connect', 3)], reduce_concat=range(2, 6)), 1: Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 1), ('skip_connect', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6)), 2: Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_3x3', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)), 4: Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('skip_connect', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6)), 6: Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6)), 0: Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6)), 5: Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))}
# Arch = {0: Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('skip_connect', 4), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 4), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6)), 1: Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 4), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 3), ('dil_conv_5x5', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6)), 2: Genotype(normal=[('sep_conv_5x5', 0), ('skip_connect', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6)), 3: Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('sep_conv_5x5', 0), ('skip_connect', 4), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6)), 4: Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('skip_connect', 4), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 3), ('sep_conv_3x3', 1), ('max_pool_3x3', 4)], reduce_concat=range(2, 6)), 5: Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 3), ('skip_connect', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6)), 6: Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('skip_connect', 4), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 4), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6)), 7: Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('skip_connect', 4), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))}
# Arch = {0: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6)), 5: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_3x3', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6)), 1: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6)), 4: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 1), ('skip_connect', 3), ('sep_conv_3x3', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 4)], reduce_concat=range(2, 6)), 3: Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 3), ('sep_conv_3x3', 0), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 2), ('skip_connect', 4)], reduce_concat=range(2, 6)), 2: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 3), ('sep_conv_3x3', 1), ('skip_connect', 4)], reduce_concat=range(2, 6)), 6: Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 3)], reduce_concat=range(2, 6)), 7: Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))}
# Arch = {0: Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2)], reduce_concat=range(2, 6)), 5: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 3), ('sep_conv_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6)), 1: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6)), 4: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)), 2: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6)), 3: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 2), ('skip_connect', 3), ('skip_connect', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)), 6: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6)), 7: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))}
#Arch = {5: Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6)), 6: Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 3), ('sep_conv_3x3', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6)), 2: Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6)), 4: Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6)), 7: Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)), 0: Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 1), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 3), ('skip_connect', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6)), 1: Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('skip_connect', 1)], reduce_concat=range(2, 6)), 3: Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))}
#Arch = {7: Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6)), 2: Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 3), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 3)], reduce_concat=range(2, 6)), 1: Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 3), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6)), 3: Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6)), 5: Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 3), ('skip_connect', 4), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6)), 6: Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('max_pool_3x3', 4), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6)), 0: Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6)), 4: Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 3), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))}
for i in range(8):
    filename_normal = "cifar10_Normal_client_11july"+str(i)
    filename_reduce = "cifar10_Reduce_client_11julys"+str(i)
    gen = Arch[i]
    plot(gen.normal, filename_normal)
    plot(gen.reduce, filename_reduce)
    #plot(Arch[i], filename)