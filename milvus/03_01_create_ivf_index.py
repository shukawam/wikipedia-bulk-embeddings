import os, time
from dotenv import load_dotenv
from pymilvus import (
    MilvusClient,
)

_ = load_dotenv()
milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
collection_name = os.getenv("COLLECTION_NAME", "Wikipedia_Ja_Collection")
index_ivf = "WIKI_IVF_INDEX"
index_hnsw = "WIKI_HNSW_INDEX"

# Milvusに接続
client = MilvusClient(milvus_uri)

def _check_old_index():
    # Indexが残っているようであれば削除
    if index_ivf in client.list_indexes(collection_name=collection_name):
        print(f"Release collection and drop {index_ivf}")
        client.release_collection(collection_name=collection_name)
        client.drop_index(collection_name=collection_name, index_name=index_ivf)
    if index_hnsw in client.list_indexes(collection_name=collection_name):
        print(f"Release collection and drop {index_hnsw}")
        client.release_collection(collection_name=collection_name)
        client.drop_index(collection_name=collection_name, index_name=index_hnsw)

def _create_index():
    index_params = client.prepare_index_params()
    # インデックスを追加
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        index_name="WIKI_IVF_INDEX",
        metric_type="COSINE"
    )
    start = time.time()
    client.create_index(collection_name=collection_name, index_params=index_params)
    print(f"IVF Index creation took {time.time() - start:.2f} seconds")

def main():
    _check_old_index()
    _create_index()
    index_info = client.describe_index(collection_name=collection_name, index_name=index_ivf)
    print(f"{index_info=}")
    client.close()

if __name__ == "__main__":
   main()
