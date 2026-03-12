import pandas as pd

def load_data(path):
    print("Loading dataset...")
    df = pd.read_csv(path)
    return df


def clean_data(df):

    print("Cleaning dataset...")

    # Drop ID column
    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)

    # Convert Total Charges to numeric
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        df["Total Charges"].fillna(df["Total Charges"].median(), inplace=True)

    return df


def save_data(df, path):
    print("Saving processed dataset...")
    df.to_csv(path, index=False)


if __name__ == "__main__":

    raw_path = "data/raw.csv"
    processed_path = "data/processed.csv"

    df = load_data(raw_path)

    df = clean_data(df)

    save_data(df, processed_path)

    print("Preprocessing completed successfully.")