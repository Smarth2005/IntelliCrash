import pandas as pd
import numpy as np
import sys
import os

def error(msg):
    print("Error:", msg)
    sys.exit(1)

def read_input_file(input_file):
    # Acceptable Input file format is either CSV or Excel
    if input_file.endswith('.xlsx'):
        return pd.read_excel(input_file)
    
    elif input_file.endswith('.csv'):
        return pd.read_csv(input_file)
    
    else:
        raise ValueError("Input file must be either .csv or .xlsx format")

def validate_inputs(input_file, weights, impacts):
    # Check if file exists before trying to read it
    if not os.path.isfile(input_file):
        error("Input file not found.") 

    try:
        df = read_input_file(input_file)
        
        if len(df.columns) < 3:
            error("Input file must contain three or more columns including the Candidates column.") 

        # Extract criteria columns and validate their dtype as numeric
        data = df.iloc[:, 1:].copy()
        for col in data.columns:
            if pd.to_numeric(data[col], errors="coerce").isnull().any():
                raise TypeError(f"Column '{col}' contains non-numeric values.") 

        # Parse and Validate Weights and Impacts
        weights = [float(w.strip()) for w in weights.split(',')]
        impacts = [i.strip() for i in impacts.split(',')]

        if len(weights) != len(data.columns) or len(impacts) != len(data.columns):
            error("Number of weights/impacts must match the number of numeric columns.") 

        if not all(i in ['+', '-'] for i in impacts):
            error("Impacts must be either + or -.")

        return df, weights, impacts

    except Exception as e:
        error(f"An error occurred: {e}") 

def topsis(input_df, weights, impacts, output_file):
    # Extract criteria columns
    data = input_df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Step 1: Vector Normalization
    rss = np.sqrt((data ** 2).sum(axis=0))
    if np.any(rss == 0):
        error("One or more criteria columns contain all zero values.")

    data = data / rss
    
    # Step 2: Calculate weighted normalized matrix
    data = data * weights
    
    # Step 3: Determine ideal best (V+) and ideal worst (V-) solutions
    n = data.shape[1]
    v_plus = np.zeros(n)
    v_minus = np.zeros(n)

    for j in range(n):
        column = data[:,j]
        max_val = np.max(column)
        min_val = np.min(column)
    
        # See if we want to maximize benefit or minimize cost (for PIS)
        if impacts[j]=='+':
            v_plus[j] = max_val
            v_minus[j] = min_val
        else:
            v_plus[j] = min_val
            v_minus[j] = max_val
    
    # Step 4: Calculate Euclidean Distances from Ideal Best and Worst
    s_plus = np.sqrt(((data - v_plus)**2).sum(axis=1))
    s_minus = np.sqrt(((data - v_minus)**2).sum(axis=1))
    
    # Step 5: Calculate Performance Score (TOPSIS Score)
    topsis_score = s_minus / (s_plus + s_minus)
    
    # Step 6: Ranking and Results DataFrame
    input_df['Topsis Score'] = topsis_score.round(4)  # Round to 4 decimal places
    input_df['Rank'] = input_df['Topsis Score'].rank(method="dense", ascending=False).astype(int)

    # Save the Output
    input_df.to_csv(output_file, index=False)
    print(f"Results successfully saved to '{output_file}'.")

def main():
    try:
        # Validate command line arguments
        if len(sys.argv) != 5:
            print("Error: Invalid number of parameters.")
            print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
            sys.exit(1)
            
        input_file = sys.argv[1]
        weights = sys.argv[2]
        impacts = sys.argv[3]
        output_file = sys.argv[4]
        
        # Validate inputs and get processed data
        df, processed_weights, processed_impacts = validate_inputs(input_file, weights, impacts)
        
        # Calculate TOPSIS
        topsis(df, processed_weights, processed_impacts, output_file)
    
    except Exception as e:
        error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()