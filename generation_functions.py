import numpy as np
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from typing import List, Dict, Any

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
        self.rules = {}
        self.dependencies = {}

    def add_independent(self, name: str, method: str, **kwargs):
        self.rules[name] = {
            'type': 'independent', 
            'method': method, 
            'kwargs': kwargs
        }
        self.dependencies[name] = []

    def add_linear_combination(self, name: str, parents: List[str], weights: List[float], bias: float, noise_std: float):
        self.rules[name] = {
            'type': 'linear',
            'parents': parents,
            'weights': weights,
            'bias': bias,
            'noise_std': noise_std
        }
        self.dependencies[name] = parents

    def add_conditional_probability(self, name: str, parent: str, condition_map: Dict[Any, Dict[str, Any]]):
        self.rules[name] = {
            'type': 'conditional',
            'parent': parent,
            'condition_map': condition_map
        }
        self.dependencies[name] = [parent]

    def _get_execution_order(self) -> List[str]:
        visited = set()
        order = []

        def dfs(node):
            if node not in visited:
                visited.add(node)
                for parent in self.dependencies.get(node, []):
                    dfs(parent)
                order.append(node)

        for col in self.rules:
            dfs(col)
        return order

    def generate_data(self, size: int) -> Dict[str, np.ndarray]:
        order = self._get_execution_order()
        data = {}

        for col in order:
            rule = self.rules[col]
            
            if rule['type'] == 'independent':
                method = getattr(self.core, rule['method'])
                data[col] = method(size=size, **rule['kwargs'])
            
            elif rule['type'] == 'linear':
                result = np.full(size, rule['bias'], dtype=float)
                for parent, weight in zip(rule['parents'], rule['weights']):
                    result += data[parent] * weight
                noise = self.core.normal(0, rule['noise_std'], size)
                data[col] = result + noise

            elif rule['type'] == 'conditional':
                parent_data = data[rule['parent']]
                result = np.empty(size, dtype=object)
                unique_vals = np.unique(parent_data)
                
                for val in unique_vals:
                    mask = parent_data == val
                    count = np.sum(mask)
                    if count > 0:
                        cond_kwargs = rule['condition_map'][val]
                        method = getattr(self.core, cond_kwargs['method'])
                        result[mask] = method(size=count, **cond_kwargs['kwargs'])
                        
                data[col] = result

        return data

class Stage3_SemanticLabels:
    def __init__(self, engine: Any):
        self.engine = engine
        self.available_columns = [
            'Age', 'Gender', 'City_Tier', 'Base_Health', 'Base_Economy', 'Random_Factor',
            'Education', 'Employment_Type', 'Housing_Status', 'Transport_Mode', 'Diet_Type', 'Marital_Status',
            'Experience_Years', 'Salary', 'Credit_Score', 'Savings', 'Debt', 'Monthly_Spend', 'Health_Index', 'Happiness_Score'
        ]

    def build_domain(self):
        self.engine.add_independent('Age', 'integers', low=18, high=66)
        self.engine.add_independent('Gender', 'choices', elements=['M', 'F'], p=[0.5, 0.5])
        self.engine.add_independent('City_Tier', 'choices', elements=['Tier1', 'Tier2', 'Tier3'], p=[0.3, 0.5, 0.2])
        self.engine.add_independent('Base_Health', 'floats', low=0.5, high=1.0)
        self.engine.add_independent('Base_Economy', 'floats', low=0.8, high=1.2)
        self.engine.add_independent('Random_Factor', 'normal', mean=0.0, std=1.0)

        self.engine.add_conditional_probability('Education', 'City_Tier', {
            'Tier1': {'method': 'choices', 'kwargs': {'elements': ['HighSchool', 'BSc', 'MSc'], 'p': [0.2, 0.5, 0.3]}},
            'Tier2': {'method': 'choices', 'kwargs': {'elements': ['HighSchool', 'BSc', 'MSc'], 'p': [0.5, 0.4, 0.1]}},
            'Tier3': {'method': 'choices', 'kwargs': {'elements': ['HighSchool', 'BSc', 'MSc'], 'p': [0.8, 0.15, 0.05]}}
        })
        self.engine.add_conditional_probability('Employment_Type', 'Gender', {
            'M': {'method': 'choices', 'kwargs': {'elements': ['Full-time', 'Part-time'], 'p': [0.8, 0.2]}},
            'F': {'method': 'choices', 'kwargs': {'elements': ['Full-time', 'Part-time'], 'p': [0.7, 0.3]}}
        })
        self.engine.add_conditional_probability('Housing_Status', 'City_Tier', {
            'Tier1': {'method': 'choices', 'kwargs': {'elements': ['Rent', 'Own'], 'p': [0.7, 0.3]}},
            'Tier2': {'method': 'choices', 'kwargs': {'elements': ['Rent', 'Own'], 'p': [0.4, 0.6]}},
            'Tier3': {'method': 'choices', 'kwargs': {'elements': ['Rent', 'Own'], 'p': [0.2, 0.8]}}
        })
        self.engine.add_conditional_probability('Transport_Mode', 'City_Tier', {
            'Tier1': {'method': 'choices', 'kwargs': {'elements': ['Public', 'Car'], 'p': [0.8, 0.2]}},
            'Tier2': {'method': 'choices', 'kwargs': {'elements': ['Public', 'Car'], 'p': [0.4, 0.6]}},
            'Tier3': {'method': 'choices', 'kwargs': {'elements': ['Public', 'Car'], 'p': [0.1, 0.9]}}
        })
        self.engine.add_conditional_probability('Diet_Type', 'Gender', {
            'M': {'method': 'choices', 'kwargs': {'elements': ['Standard', 'Vegetarian'], 'p': [0.85, 0.15]}},
            'F': {'method': 'choices', 'kwargs': {'elements': ['Standard', 'Vegetarian'], 'p': [0.7, 0.3]}}
        })
        self.engine.add_conditional_probability('Marital_Status', 'City_Tier', {
            'Tier1': {'method': 'choices', 'kwargs': {'elements': ['Single', 'Married'], 'p': [0.6, 0.4]}},
            'Tier2': {'method': 'choices', 'kwargs': {'elements': ['Single', 'Married'], 'p': [0.4, 0.6]}},
            'Tier3': {'method': 'choices', 'kwargs': {'elements': ['Single', 'Married'], 'p': [0.2, 0.8]}}
        })

        self.engine.add_linear_combination('Experience_Years', ['Age'], [0.8], -14.0, 2.0)
        self.engine.add_linear_combination('Salary', ['Experience_Years', 'Base_Economy'], [3000.0, 20000.0], 10000.0, 5000.0)
        self.engine.add_linear_combination('Credit_Score', ['Salary', 'Age'], [0.002, 1.5], 300.0, 30.0)
        self.engine.add_linear_combination('Savings', ['Salary', 'Base_Health'], [0.2, 5000.0], -2000.0, 1000.0)
        self.engine.add_linear_combination('Debt', ['Salary'], [0.4], 5000.0, 2000.0)
        self.engine.add_linear_combination('Monthly_Spend', ['Salary', 'Credit_Score'], [0.3, 2.0], 1000.0, 500.0)
        self.engine.add_linear_combination('Health_Index', ['Base_Health', 'Age'], [100.0, -0.5], 0.0, 5.0)
        self.engine.add_linear_combination('Happiness_Score', ['Salary', 'Health_Index'], [0.0001, 0.5], 20.0, 5.0)

    def generate_table(self, user_columns: List[str], size: int) -> Dict[str, np.ndarray]:
        all_data = self.engine.generate_data(size)
        result = {}
        for col in user_columns:
            if col in all_data:
                result[col] = all_data[col]
        return result