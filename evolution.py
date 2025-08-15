import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from .models.cnn import CNNGenome, make_default_genome as make_default_cnn
from .models.transformer import TransGenome, make_default_genome as make_default_trans

@dataclass
class Individual:
    genome: Any
    score: float = -1e9

def init_population(model_type:str, size_label:str, pop_size:int, in_ch:int, n_classes:int, img_size:int=32):
    pop = []
    for _ in range(pop_size):
        if model_type=="cnn":
            g = make_default_cnn(size_label, in_ch, n_classes)
        else:
            g = make_default_trans(size_label, img_size, in_ch, n_classes)
        pop.append(Individual(genome=g))
    return pop

def mutate(genome, model_type:str):
    g = deepcopy(genome)
    if model_type=="cnn":
        # choose mutation
        op = random.choice(["add","remove","widen","kernel"])
        if op=="add":
            pos = random.randint(0, len(g.blocks))
            g.blocks.insert(pos, (random.choice([32,64,128,256]), random.choice([3,3,5]), random.choice([1,1,2])))
        elif op=="remove" and len(g.blocks)>1:
            pos = random.randrange(len(g.blocks))
            g.blocks.pop(pos)
        elif op=="widen":
            i = random.randrange(len(g.blocks))
            oc,k,s = g.blocks[i]
            g.blocks[i]=(int(oc*random.choice([1.25,1.5])) , k, s)
        elif op=="kernel":
            i = random.randrange(len(g.blocks))
            oc,k,s = g.blocks[i]
            g.blocks[i]=(oc, random.choice([3,5]), s)
    else:
        op = random.choice(["depth","dim","heads","mlp"])
        if op=="depth":
            g.depth = max(2, g.depth + random.choice([-1,1]))
        elif op=="dim":
            g.dim = max(64, int(g.dim * random.choice([0.75, 1.25])))
        elif op=="heads":
            g.heads = max(2, min(12, g.heads + random.choice([-2,-1,1,2])))
        elif op=="mlp":
            g.mlp_dim = max(64, int(g.mlp_dim * random.choice([0.75, 1.25])))
    return g

def crossover(g1, g2, model_type:str):
    import random
    if model_type=="cnn":
        a = g1.blocks
        b = g2.blocks
        if not a or not b: return deepcopy(g1)
        cut_a = random.randrange(1, len(a))
        cut_b = random.randrange(1, len(b))
        child_blocks = a[:cut_a] + b[cut_b:]
        child = deepcopy(g1)
        child.blocks = child_blocks
        return child
    else:
        child = deepcopy(g1)
        # 1-point crossover over tuple of attrs
        attrs = ["patch","dim","depth","heads","mlp_dim","pool"]
        cut = random.randrange(1, len(attrs))
        for i, name in enumerate(attrs):
            source = g1 if i<cut else g2
            setattr(child, name, getattr(source, name))
        return child
