
from collections import namedtuple
from visualize import plot
import genotypes

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

FedNAS_v1 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)


plot(FedNAS_v1, 'normal')
