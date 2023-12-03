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

PRIMITIVES_Rethink = [
    'none',
    'noise',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
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

######## S1-S4 Space ########
#### cifar10 s1 - s4
darts_pt_s1_1 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6)) # 1.87
darts_pt_s1_2 = Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 3), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6)) # 2.02
darts_pt_s2_1 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 3.09
darts_pt_s2_2 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 3), ('skip_connect', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6)) # 2.79
darts_pt_s3_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 3), ('skip_connect', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6)) # 3.42
darts_pt_s3_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 3)], reduce_concat=range(2, 6)) # 3.54
darts_pt_s4_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)) # 4.7
darts_pt_s4_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6)) # 4.7

blank_pt_s1_1 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('skip_connect', 3)], reduce_concat=range(2, 6)) # 2.36
blank_pt_s1_2 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6)) # 2.39
blank_pt_s2_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 2), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 3.45
blank_pt_s2_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6)) # 4.3
blank_pt_s3_1 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6)) # 2.9
blank_pt_s3_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6)) # 3.81
blank_pt_s4_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)) # 4.7
blank_pt_s4_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)) # 4.7

#### cifar100 s1 - s4
darts_pt_s1_c100_1 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6)) # 2.47
darts_pt_s1_c100_2 = Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 3), ('skip_connect', 2), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 2.07
darts_pt_s2_c100_1 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 3), ('skip_connect', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 3), ('sep_conv_3x3', 2), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 3.08
darts_pt_s2_c100_2 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('sep_conv_3x3', 3), ('skip_connect', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 3), ('sep_conv_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6)) # 3.11
darts_pt_s3_c100_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6)) # 3.83
darts_pt_s3_c100_2 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 3), ('sep_conv_3x3', 2), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 3.44
darts_pt_s4_c100_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)) # 4.75
darts_pt_s4_c100_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6)) # 4.75

blank_pt_s1_c100_1 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 2.46
blank_pt_s1_c100_2 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('skip_connect', 3)], reduce_concat=range(2, 6)) # 2.44
blank_pt_s2_c100_1 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 3.14
blank_pt_s2_c100_2 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 2), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 2.84
blank_pt_s3_c100_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6)) # 3.89
blank_pt_s3_c100_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 3), ('skip_connect', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)) # 3.95
blank_pt_s4_c100_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)) # 4.75
blank_pt_s4_c100_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)) # 4.75

#### svhn s1 - s4
darts_pt_s1_svhn_1 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('skip_connect', 2), ('dil_conv_5x5', 3), ('avg_pool_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 2.38
darts_pt_s1_svhn_2 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('dil_conv_3x3', 3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 2.05
darts_pt_s2_svhn_1 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 2), ('sep_conv_3x3', 3), ('skip_connect', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6)) # 3.08
darts_pt_s2_svhn_2 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6)) # 3.14
darts_pt_s3_svhn_1 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6)) # 3.50
darts_pt_s3_svhn_2 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('skip_connect', 2), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6)) # 2.82
darts_pt_s4_svhn_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6)) # 4.70
darts_pt_s4_svhn_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6)) # 4.70

blank_pt_s1_svhn_1 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 2), ('skip_connect', 3), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('dil_conv_5x5', 3), ('skip_connect', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6)) # 2.95
blank_pt_s1_svhn_2 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6)) # 3.43
blank_pt_s2_svhn_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('skip_connect', 2), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 4.20
blank_pt_s2_svhn_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6)) # 4.26
blank_pt_s3_svhn_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('skip_connect', 2), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6)) # 3.90
blank_pt_s3_svhn_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6)) # 4.64
blank_pt_s4_svhn_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6)) # 4.70
blank_pt_s4_svhn_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6)) # 4.70




######## DARTS Space ########
##### darts
darts_pt_s5_0 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 2), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 2.85
darts_pt_s5_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_conv_3x3', 3), ('sep_conv_5x5', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6)) # 3.05
darts_pt_s5_2 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 3), ('max_pool_3x3', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('skip_connect', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6)) # 2.66
darts_pt_s5_3 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 1), ('max_pool_3x3', 3), ('sep_conv_5x5', 2), ('skip_connect', 3)], reduce_concat=range(2, 6)) # 3.33

#### blank
blank_pt_s5_0 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 0), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6)) # 3.04
blank_pt_s5_1 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 2), ('sep_conv_3x3', 3), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 2), ('max_pool_3x3', 3), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6)) # 3.15
blank_pt_s5_2 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 3), ('max_pool_3x3', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6)) # 2.58
