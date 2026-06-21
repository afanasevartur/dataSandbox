import pandas as pd
from generation_functions import Stage1_CoreGenerator, Stage2_MathEngine, Stage3_SemanticLabels

if __name__ == "__main__":
    
    print("--- initialize engine ---")
    core = Stage1_CoreGenerator(seed=42)
    math_engine = Stage2_MathEngine(core)
    domain = Stage3_SemanticLabels(math_engine)
    
    domain.build_domain()
    
    print("\n--- available columns ---")
    print(", ".join(domain.available_columns))
    
    print("\n--- user execution ---")
    
    user_selected_columns = [
        'Age', 'Gender', 'City_Tier', 'Education', 
        'Experience_Years', 'Salary', 'Credit_Score', 
        'Housing_Status', 'Monthly_Spend'
    ]
    
    target_rows = 1000
    
    print(f"Generating {target_rows} rows...")
    output_data = domain.generate_table(user_columns=user_selected_columns, size=target_rows)
    
    df = pd.DataFrame(output_data)
    
    output_filename = "synthetic_dataset.csv"
    df.to_csv(output_filename, index=False)
    
    print(f"\nSuccess. File saved as: {output_filename}")
    print("\nData preview:")
    print(df.head())