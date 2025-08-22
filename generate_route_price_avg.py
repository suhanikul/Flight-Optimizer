# generate_route_avg_prices.py
import pandas as pd
import json

df = pd.read_csv("data/goibibo_flights_data.csv")
df["price"] = df["price"].astype(str).str.replace(",", "").astype(float)

df = df[df["price"] > 200]  # clean
route_avg_prices = (
    df.groupby([df["from"].str.lower(), df["to"].str.lower()])["price"]
    .mean()
    .reset_index()
)

route_avg_prices_dict = {
    f"{row['from']}->{row['to']}": row["price"]
    for _, row in route_avg_prices.iterrows()
}

with open("model/route_avg_prices.json", "w") as f:
    json.dump(route_avg_prices_dict, f, indent=2)

print(f"Saved average prices for {len(route_avg_prices_dict)} routes.")
