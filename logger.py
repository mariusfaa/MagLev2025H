import csv
import json
import os
import types  # <--- NY: Trengs for å identifisere moduler
import config
import numpy as np

class SimulationLogger:
    def __init__(self, filename_prefix):
        self.filename_prefix = filename_prefix
        self.data = []
        self.params = {}
        
        # --- NY: Automatisk opprettelse av mappe ---
        directory = os.path.dirname(filename_prefix)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Opprettet mappe: {directory}")
            except OSError as e:
                print(f"Kunne ikke opprette mappe {directory}: {e}")
        # -------------------------------------------

    def log(self, time, position, velocity, target, force, noise, estimated_pos=None):
        """Logger ett tidssteg."""
        entry = {
            "Time": round(time, 4),
            "Position": position,
            "Velocity": velocity,
            "Target": target,
            "Force": force,
            "Noise": noise
        }
        if estimated_pos is not None:
            entry["Estimated_Position"] = estimated_pos
            
        self.data.append(entry)

    def save(self, extra_params=None):
        """
        Lagrer data til CSV og parametere til JSON.
        extra_params: En dict med spesifikke parametere for kontrolleren.
        """
        # 1. Samle parametere fra config, men filtrer bort moduler (som numpy) og funksjoner
        all_params = {
            k: v for k, v in vars(config).items() 
            if not k.startswith("__") 
            and not callable(v)
            and not isinstance(v, types.ModuleType)  # <--- NY: Ignorerer importerte moduler
        }
        
        if extra_params:
            all_params.update(extra_params)

        # Lagre Parametere (JSON)
        param_filename = f"{self.filename_prefix}_params.json"
        
        # Definer en trygg serialiseringsfunksjon
        def json_serializer(obj):
            if hasattr(obj, 'dtype'): # Håndterer numpy-tall (float32 osv)
                return float(obj)
            if isinstance(obj, np.ndarray): # Håndterer numpy-arrays
                return obj.tolist()
            return str(obj) # Fallback for alt annet

        try:
            with open(param_filename, 'w') as f:
                json.dump(all_params, f, indent=4, default=json_serializer)
            print(f"Parametere lagret til: {param_filename}")
        except Exception as e:
            print(f"Feil ved lagring av parametere: {e}")

        # Lagre Simuleringsdata (CSV)
        csv_filename = f"{self.filename_prefix}_data.csv"
        if not self.data:
            print("Ingen data å lagre.")
            return

        keys = self.data[0].keys()
        try:
            with open(csv_filename, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.data)
            print(f"Data lagret til: {csv_filename}")
        except Exception as e:
             print(f"Feil ved lagring av CSV: {e}")