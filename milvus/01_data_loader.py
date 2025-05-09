import os
import pandas as pd
from dotenv import load_dotenv
from pymilvus import (
    CollectionSchema,
    MilvusClient,
    DataType,
)

_ = load_dotenv()
milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
collection_name = os.getenv("COLLECTION_NAME", "Wikipedia_Ja_Collection")
csv_file = os.getenv("CSV_FILE", "./data/wiki_ja_embeddings_2025-04-01.csv")

# Milvusに接続
client = MilvusClient(milvus_uri)

def _add_schema(schema: CollectionSchema) -> None:
    print("Adding schema to collection...")
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(field_name="pageid", datatype=DataType.INT64)
    schema.add_field(field_name="revid", datatype=DataType.INT64)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)
    
def _create_collection():
    # コレクションの作成
    if client.has_collection(collection_name):
        print(f"Collection {collection_name} already exists.")
        client.drop_collection(collection_name=collection_name)
    schema = client.create_schema(
        auto_id=False,
        enabled_dynamic_field=False
    )
    _add_schema(schema)
    # コレクションの作成                
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )
    load_state = client.get_load_state(collection_name=collection_name)
    print(f"{load_state=}")

def _load_data():
    print("Loading data into collection...")
    chunk_size = 10_000
    total_rows_inserted = 0
    try:
        for chunk_num, chunk_df in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
            print(f"Processing chunk {chunk_num + 1} ({len(chunk_df)} rows)...")
            print(f"{chunk_df.head()=}")
            if 'embedding' in chunk_df.columns and isinstance(chunk_df['embedding'].iloc[0], str):
                try:
                    import json # ast.literal_eval より安全な場合がある
                    chunk_df['embedding'] = chunk_df['embedding'].apply(json.loads)
                except Exception as e:
                    print(f"Warning: Could not parse 'embedding' column as JSON list in chunk {chunk_num + 1}. Error: {e}")
                    print("Ensure the 'embedding' column contains valid list-like strings, e.g., '[0.1, 0.2, ...]' or use appropriate parsing.")
                    # ここで処理を続けるか、エラーとして停止するかを決定
                    # return
            data = chunk_df.to_dict(orient="records")
            if not data:
                print(f"Chunk {chunk_num + 1} is empty. Skipping...")
                continue
            print(f"Chunk {chunk_num + 1} rows into {collection_name}...")
            result = client.insert(collection_name=collection_name, data=data)
            inserted_count = len(result.keys())
            print(f"Inserted {inserted_count} rows into {collection_name}.")
            total_rows_inserted += inserted_count
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    if total_rows_inserted > 0:
        client.flush(collection_name=collection_name)

def main():
    _create_collection()
    _load_data()
    print("Data loading completed.")
    client.close()

if __name__ == "__main__":
    print("Starting Milvus data loader...")
    main()
