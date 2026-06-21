from generation_functions import CoreGenerator

if __name__ == "__main__":
    print("--- fixed seed ---")
    test_engine = CoreGenerator(seed=42)
    print("Age:", test_engine.generate_integers(18, 65, size=5))
    print("Status:", test_engine.generate_choices(["Active", "Closed"], size=5, probabilities=[0.8, 0.2]))

    print("\n--- no seed ---")
    prod_engine = CoreGenerator()
    print("Age:", prod_engine.generate_integers(18, 65, size=5))
    print("Status:", prod_engine.generate_choices(["Active", "Closed"], size=5, probabilities=[0.8, 0.2]))