import pandas as pd
import numpy as np
import glob
import os

def calculate_rmse(directory="."):
    """
    Søker etter CSV-filer i gitt mappe og beregner RMSE mellom referanse og posisjon.
    """
    # Finn alle csv-filer i mappen
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not csv_files:
        print(f"Ingen CSV-filer funnet i mappen: {os.path.abspath(directory)}")
        return

    print(f"\n{'Filnavn':<50} | {'RMSE':<20}")
    print("-" * 75)

    results = []

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        
        try:
            # Les CSV-filen
            df = pd.read_csv(filepath)
            
            # --- KONFIGURASJON AV KOLONNENAVN ---
            # Oppdatert liste som inkluderer 'Target' og 'Position' (store bokstaver)
            possible_ref_cols = ['Target', 'reference', 'ref', 'target', 'target_height', 'Target Height', 'r']
            possible_pos_cols = ['Position', 'position', 'pos', 'height', 'ball_height', 'y', 'Height']

            # Finn første match i kolonnene
            ref_col = next((c for c in df.columns if c in possible_ref_cols), None)
            pos_col = next((c for c in df.columns if c in possible_pos_cols), None)

            if ref_col and pos_col:
                # Hent ut data som numpy arrays
                ref_data = df[ref_col].values
                pos_data = df[pos_col].values

                # --- BEREGN RMSE ---
                # Formel: sqrt( mean( (ref - pos)^2 ) )
                mse = np.mean((ref_data - pos_data) ** 2)
                rmse = np.sqrt(mse)

                print(f"{filename:<50} | {rmse:.5f}")
                results.append((filename, rmse))
            
            else:
                print(f"{filename:<50} | FEIL: Fant ikke kolonner")
                print(f"   Tilgjengelige kolonner: {list(df.columns)}")

        except Exception as e:
            print(f"{filename:<50} | ERROR: {str(e)}")

    return results

if __name__ == "__main__":
    calculate_rmse("benchmark_results/20251206_153610/sine_Period_1")  # Endre til ønsket mappe