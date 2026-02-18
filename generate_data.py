import pandas as pd
import numpy as np

def generate_synthetic_data(num_records=1000, output_file='dataset.csv'):
    np.random.seed(42)
    
    data = {
        'User_ID': np.arange(1, num_records + 1),
        'Age': np.random.randint(18, 61, num_records),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], num_records),
        'Occupation': np.random.choice(['Student', 'Professional', 'Freelancer', 'Business Owner'], num_records),
        'Device_Type': np.random.choice(['Android', 'iOS'], num_records),
        'Daily_Phone_Hours': np.random.uniform(1, 12, num_records),
        'Social_Media_Hours': np.random.uniform(0.5, 8, num_records),
        'Work_Productivity_Score': np.random.randint(1, 11, num_records),
        'Sleep_Hours': np.random.uniform(3, 10, num_records),
        'App_Usage_Count': np.random.randint(5, 50, num_records),
        'Caffeine_Intake_Cups': np.random.randint(0, 6, num_records),
        'Weekend_Screen_Time_Hours': np.random.uniform(2, 14, num_records)
    }
    
    df = pd.DataFrame(data)
    
    # Generate Stress_Level with some correlation to features
    # Stress increases with phone usage, social media, caffeine, and decreases with sleep and productivity
    base_stress = 5
    stress_factors = (
        0.2 * (df['Daily_Phone_Hours'] - 6) +
        0.3 * (df['Social_Media_Hours'] - 4) -
        0.4 * (df['Work_Productivity_Score'] - 5) -
        0.5 * (df['Sleep_Hours'] - 7) +
        0.3 * (df['Caffeine_Intake_Cups'] - 2) +
        np.random.normal(0, 1.5, num_records) # Noise
    )
    
    df['Stress_Level'] = base_stress + stress_factors
    df['Stress_Level'] = df['Stress_Level'].clip(1, 10).round().astype(int)
    
    # Ensure Social_Media_Hours <= Daily_Phone_Hours (approx)
    df['Social_Media_Hours'] = np.minimum(df['Social_Media_Hours'], df['Daily_Phone_Hours'])
    
    print(f"Generating {num_records} records...")
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
