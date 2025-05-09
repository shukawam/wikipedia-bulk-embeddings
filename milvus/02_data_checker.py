import os
from dotenv import load_dotenv
from pymilvus import (
    MilvusClient,
)

_ = load_dotenv()
milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
collection_name = os.getenv("COLLECTION_NAME", "Wikipedia_Ja_Collection")

# Milvusに接続
client = MilvusClient(milvus_uri)

def main():
    if client.has_collection(collection_name):
        print(f"Collection {collection_name} already exists.")
    else:
        print(f"Collection {collection_name} does not exist. Creating collection using 01_data_loader.py...")
        return
    collection_list = client.list_collections()
    print(f"{collection_list=}")
    row_count = client.get_collection_stats(collection_name=collection_name)
    print(f"{row_count=}")
    client.close()

if __name__ == "__main__":
    main()
