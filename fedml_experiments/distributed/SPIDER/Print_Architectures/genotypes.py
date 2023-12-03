from collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'skip_connect',
    'max_pool_3x3',
    'avg_pool_3x3',
    'none',
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

FedNAS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 3), ('dil_conv_5x5', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

FedNAS_debug = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0),
                 ('skip_connect', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 4)],
         normal_concat=range(2, 6), reduce=None, reduce_concat=None)



C1 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)

C2 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)

C3 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)

C4 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)

C5 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)

C6 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)

C7 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)

C8 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)

C9 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)

C10 =Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)


Global = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)
GG = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)
CC2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)
CC3 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)
CC4 = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 0), ('sep_conv_3x3', 3), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)
CC1 = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=None, reduce_concat=None)