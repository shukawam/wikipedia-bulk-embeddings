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
    before_state = client.get_load_state(collection_name=collection_name)
    print(f"{before_state=}")
    # メモリ上にコレクションをロード
    client.load_collection(collection_name=collection_name)
    after_state = client.get_load_state(collection_name=collection_name)
    print(f"{after_state=}")
    

if __name__ == "__main__":
    main()
