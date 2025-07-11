import psycopg2
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import pandas as pd

load_dotenv()

MATERIAL_EMISSIONS = {
    "clothing": {
        "cotton": 5.9,
        "polyester": 9.5,
        "leather": 65.0,
        "faux leather": 15.8
    },
    "kitchenware": {
        "stainless steel": 6.15,
        "aluminum": 18.5,
        "polypropylene": 2.3,
        "silicone": 3.28,
        "glass": 2.25
    }
}

def get_db_connection():
    return psycopg2.connect(os.getenv("DB_URL"))

def parse_composition(comp_str):
    vec = {}
    for item in comp_str.split(','):
        mat, perc = item.split(':')
        vec[mat.strip().lower()] = float(perc)
    return vec

def composition_vector(comp_str, allowed_materials):
    mat_perc = parse_composition(comp_str)
    vec = [mat_perc.get(mat, 0) / 100 for mat in allowed_materials]
    return vec

def fetch_data(category):
    conn = get_db_connection()
    cur = conn.cursor()
    
    if category == "clothing":
        cur.execute("SELECT id, product_name, subcategory, material_composition, ghg_emission FROM clothing;")
        cols = ["id", "product_name", "subcategory", "material_composition", "ghg_emission"]
    elif category == "kitchenware":
        cur.execute("SELECT id, product_name, subcategory, material_composition, ghg_emission FROM kitchenware;")
        cols = ["id", "product_name", "subcategory", "material_composition", "ghg_emission"]
    else:
        raise ValueError(f"Unsupported category: {category}")

    rows = cur.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=cols)

def recommend_knn(category, selected_id, k=3):
    if category not in MATERIAL_EMISSIONS:
        raise ValueError(f"No material emission data for category: {category}")

    df = fetch_data(category)
    allowed_materials = list(MATERIAL_EMISSIONS[category].keys())

    # One-hot encode subcategory
    ohe = OneHotEncoder()
    subcat_encoded = ohe.fit_transform(df[["subcategory"]]).toarray()

    mat_vectors = np.array([composition_vector(c, allowed_materials) for c in df["material_composition"]])

    # Combine features: subcat + materials + emission
    features = np.hstack([subcat_encoded, mat_vectors, df[["ghg_emission"]].values])

    if selected_id not in df["id"].values:
        print(" Selected item ID not found.")
        return

    knn = NearestNeighbors(n_neighbors=k+1)
    knn.fit(features)

    idx = df[df["id"] == selected_id].index[0]
    distances, indices = knn.kneighbors([features[idx]])

    print(f"\n Selected Item: {df.iloc[idx]['product_name']} (GHG: {df.iloc[idx]['ghg_emission']} kg CO2e)\n")
    print(" Recommended Greener Alternatives:\n")

    for i in indices[0]:
        if i == idx:
            continue
        item = df.iloc[i]
        if item["ghg_emission"] < df.iloc[idx]["ghg_emission"]:
            clean_name = item['product_name'].replace('\u2011', '-')
            print(f"{clean_name} | GHG: {item['ghg_emission']} kg CO2e")

recommend_knn(category="kitchenware", selected_id=1, k=3)
