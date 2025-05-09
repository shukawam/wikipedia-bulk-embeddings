import os, time
from typing import List
from dotenv import load_dotenv
from pymilvus import (
    MilvusClient,
)
from oci.auth.signers.instance_principals_security_token_signer import InstancePrincipalsSecurityTokenSigner
from oci.generative_ai_inference.generative_ai_inference_client import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode
)

_ = load_dotenv()
milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
collection_name = os.getenv("COLLECTION_NAME", "Wikipedia_Ja_Collection")
compartment_id = os.getenv("COMPARTMENT_ID", "")
output_fields = ["id", "pageid", "revid", "title", "section", "text"]

# Milvusに接続
client = MilvusClient(milvus_uri)
# Generative AIサービスのクライアントを初期化
genai_client = GenerativeAiInferenceClient(
    config={},
    signer=InstancePrincipalsSecurityTokenSigner(),
    service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com"
)

def _load_collection():
    client.load_collection(collection_name=collection_name)
    state = client.get_load_state(collection_name=collection_name)
    print(f"{state=}")

def _get_query_embedding(query: str) -> List[float]:
    res = genai_client.embed_text(
        embed_text_details=EmbedTextDetails(
            inputs=[query],
            serving_mode=OnDemandServingMode(
                model_id="cohere.embed-multilingual-v3.0",
            ),
            compartment_id=compartment_id,
            input_type="SEARCH_QUERY"
        )
    )
    return res.data.embeddings

def _similarity_search(vector: List[float]) -> List[List[dict]]:
    res = client.search(
        collection_name=collection_name,
        data=vector,
        output_fields=output_fields
    )
    return res

def main():
    _load_collection()
    query = "サヴォワ地方はどこの国？"
    query_vector = _get_query_embedding(query)
    start = time.time()
    search_result = _similarity_search(query_vector)
    print(f"Vector search took {time.time() - start} seconds")
    print(search_result)

if __name__ == "__main__":
    main()
