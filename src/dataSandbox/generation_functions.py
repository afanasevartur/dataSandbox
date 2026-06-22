import numpy as np
import pandas as pd
from typing import List, Dict, Any

class Rule:
    pass

class Independent(Rule):
    def __init__(self, method: str, **kwargs):
        self.type = 'independent'
        self.method = method
        self.kwargs = kwargs

class Linear(Rule):
    def __init__(self, parents_weights: Dict[str, float], bias: float = 0.0, noise_std: float = 0.0):
        self.type = 'linear'
        self.parents = list(parents_weights.keys())
        self.weights = list(parents_weights.values())
        self.bias = bias
        self.noise_std = noise_std

class Conditional(Rule):
    def __init__(self, parent: str, condition_map: Dict[Any, Independent]):
        self.type = 'conditional'
        self.parent = parent
        self.condition_map = condition_map

class Stage1_CoreGenerator:
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        return self.rng.integers(low, high, size=size)

    def floats(self, low: float, high: float, size: int) -> np.ndarray:
        return self.rng.uniform(low, high, size=size)

    def normal(self, mean: float, std: float, size: int) -> np.ndarray:
        return self.rng.normal(loc=mean, scale=std, size=size)

    def choices(self, elements: List[Any], size: int, p: List[float] = None) -> np.ndarray:
        return self.rng.choice(elements, size=size, p=p)

class Stage2_MathEngine:
    def __init__(self, core: Stage1_CoreGenerator):
        self.core = core
        self.schema = {}
        self.execution_order = []

    def build_schema(self, schema: Dict[str, Rule]):
        self.schema = schema
        self._determine_execution_order()

    def _determine_execution_order(self):
        graph = {}
        in_degree = {col: 0 for col in self.schema}
        
        for col, rule in self.schema.items():
            deps = []
            if rule.type == 'linear':
                deps = rule.parents
            elif rule.type == 'conditional':
                deps = [rule.parent]
                
            for dep in deps:
                if dep not in self.schema:
                    raise ValueError(f"Column '{dep}' assigned as parent for '{col}', but it's disappeared!")
                if dep not in graph:
                    graph[dep] = []
                graph[dep].append(col)
                in_degree[col] += 1
                
        queue = [col for col, deg in in_degree.items() if deg == 0]
        self.execution_order = []
        
        while queue:
            curr = queue.pop(0)
            self.execution_order.append(curr)
            for neighbor in graph.get(curr, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(self.execution_order) != len(self.schema):
            raise ValueError("Error in hiererchy!")

    def generate_dataframe(self, size: int) -> pd.DataFrame:
        data = {}
        
        for col in self.execution_order:
            rule = self.schema[col]
            
            if rule.type == 'independent':
                method = getattr(self.core, rule.method)
                data[col] = method(size=size, **rule.kwargs)
            
            elif rule.type == 'linear':
                result = np.full(size, rule.bias, dtype=float)
                for parent, weight in zip(rule.parents, rule.weights):
                    result += data[parent] * weight
                noise = self.core.normal(0, rule.noise_std, size)
                data[col] = result + noise
            
            elif rule.type == 'conditional':
                parent_data = data[rule.parent]
                result = np.empty(size, dtype=object)
                unique_vals = np.unique(parent_data)
                
                for val in unique_vals:
                    mask = parent_data == val
                    count = np.sum(mask)
                    if count > 0:
                        cond_rule = rule.condition_map[val]
                        method = getattr(self.core, cond_rule.method)
                        result[mask] = method(size=count, **cond_rule.kwargs)
                        
                data[col] = result
                
        return pd.DataFrame(data)[list(self.schema.keys())]
