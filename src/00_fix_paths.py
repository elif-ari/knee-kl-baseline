import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

def fix(csv_name):
    csv_path = project_root / "data" / csv_name
    df = pd.read_csv(csv_path)

    def to_relative(p):
        p = Path(p)
        parts = list(p.parts)
        if "data" in parts:
            i = parts.index("data")
            rel = Path(*parts[i:])
            return str(rel).replace("\\", "/")
        return str(p).replace("\\", "/")

    df["path"] = df["path"].apply(to_relative)
    df.to_csv(csv_path, index=False)

    print(f"{csv_name} updated.")
    print("Example row:", df.iloc[0].to_dict())

if __name__ == "__main__":
    fix("train.csv")
    fix("val.csv")
    fix("test.csv")