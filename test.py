from dataSandbox.generation_functions import Stage1_CoreGenerator, Stage2_MathEngine
from dataSandbox.generation_functions import Independent, Linear, Conditional

if __name__ == "__main__":
    
    generation_schema = {
        'Age': Independent('integers', low=18, high=66),
        'Gender': Independent('choices', elements=['M', 'F'], p=[0.5, 0.5]),
        'City_Tier': Independent('choices', elements=['Tier1', 'Tier2', 'Tier3'], p=[0.3, 0.5, 0.2]),
        'Base_Economy': Independent('floats', low=0.8, high=1.2),
        
        'Education': Conditional(parent='City_Tier', condition_map={
            'Tier1': Independent('choices', elements=['HighSchool', 'BSc', 'MSc'], p=[0.2, 0.5, 0.3]),
            'Tier2': Independent('choices', elements=['HighSchool', 'BSc', 'MSc'], p=[0.5, 0.4, 0.1]),
            'Tier3': Independent('choices', elements=['HighSchool', 'BSc', 'MSc'], p=[0.8, 0.15, 0.05])
        }),
        
        'Housing_Status': Conditional(parent='City_Tier', condition_map={
            'Tier1': Independent('choices', elements=['Rent', 'Own'], p=[0.7, 0.3]),
            'Tier2': Independent('choices', elements=['Rent', 'Own'], p=[0.4, 0.6]),
            'Tier3': Independent('choices', elements=['Rent', 'Own'], p=[0.2, 0.8])
        }),
        
        'Experience_Years': Linear({'Age': 0.8}, bias=-14.0, noise_std=2.0),
        'Salary': Linear({'Experience_Years': 3000.0, 'Base_Economy': 20000.0}, bias=10000.0, noise_std=5000.0),
        'Credit_Score': Linear({'Salary': 0.002, 'Age': 1.5}, bias=300.0, noise_std=30.0),
        'Monthly_Spend': Linear({'Salary': 0.3, 'Credit_Score': 2.0}, bias=1000.0, noise_std=500.0)
    }

    target_rows = 1000
    output_filename = "synthetic_dataset.csv"

    core = Stage1_CoreGenerator(seed=42)
    math_engine = Stage2_MathEngine(core)
    
    math_engine.build_schema(generation_schema)
    
    df = math_engine.generate_dataframe(size=target_rows)
    df.to_csv(output_filename, index=False)
    
    print(df.head())