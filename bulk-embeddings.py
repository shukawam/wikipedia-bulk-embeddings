import multiprocessing as mp
import threading, csv, logging, os, datetime, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import (
    load_dataset,
    DatasetDict,
    Dataset,
    IterableDataset,
    IterableDatasetDict,
)
from oci.retry import ExponentialBackoffWithFullJitterRetryStrategy
from oci.retry.retry_checkers import RetryCheckerContainer, LimitBasedRetryChecker
from oci.config import from_file
from oci.auth.signers import InstancePrincipalsSecurityTokenSigner
from oci.generative_ai_inference.generative_ai_inference_client import (
    GenerativeAiInferenceClient,
)
from oci.generative_ai_inference.models import EmbedTextDetails, OnDemandServingMode

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)-9s  %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 96
MAX_THREADS = min(mp.cpu_count(), 2)
OUTPUT_FILE = f"wiki_ja_embeddings_{datetime.date.today()}.csv"
CSV_COLUMNS = ["id", "pageid", "revid", "title", "section", "text", "embedding"]
LOCK = threading.Lock()

# 環境に応じて設定
COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")
if COMPARTMENT_ID == None:
    logger.info("compartment_idを設定してください")
    sys.exit(1)
REGION = os.getenv("REGION", "us-chicago-1")
USE_IP = os.getenv("USE_IP", False)

checker_container = RetryCheckerContainer(checkers=[LimitBasedRetryChecker()])
retry_strategy = ExponentialBackoffWithFullJitterRetryStrategy(
    base_sleep_time_seconds=30,
    exponent_growth_factor=4,
    max_wait_between_calls_seconds=60,
    checker_container=checker_container,
)

if USE_IP == False:
    logger.info("Use default config")
    generative_ai_client = GenerativeAiInferenceClient(
        config=from_file(),
        service_endpoint=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
        retry_strategy=retry_strategy,
    )
else:
    logger.info("Use Instance Principal")
    generative_ai_client = GenerativeAiInferenceClient(
        config={},
        signer=InstancePrincipalsSecurityTokenSigner(),
        service_endpoint=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
        retry_strategy=retry_strategy,
    )

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(CSV_COLUMNS)


def get_text_embeddings(texts: list[str]) -> list[list[float]]:
    try:
        response = generative_ai_client.embed_text(
            EmbedTextDetails(
                inputs=texts,
                serving_mode=OnDemandServingMode(
                    model_id="cohere.embed-multilingual-v3.0"
                ),
                compartment_id=COMPARTMENT_ID,
                input_type="SEARCH_DOCUMENT",
            )
        )
        return response.data.embeddings
    except Exception as e:
        logger.error(f"Embeddingの取得に失敗しました: {e}")
        return [None] * len(texts)


def batch_processing(batch):
    ids = batch["id"]
    page_ids = batch["pageid"]
    revids = batch["revid"]
    titles = batch["title"]
    sections = batch["section"]
    texts = batch["text"]
    embeddings = get_text_embeddings(texts)
    # OOM(Out-of-Memory)が発生しないようにバッチ処置単位でロックをとりながらCSVに書き出し
    with LOCK:
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for i in range(len(batch)):
                writer.writerow(
                    [
                        ids[i],
                        page_ids[i],
                        revids[i],
                        titles[i],
                        sections[i],
                        texts[i],
                        embeddings[i],
                    ]
                )


def load_wikipedia_japanese_datasets() -> (
    DatasetDict | Dataset | IterableDatasetDict | IterableDataset
):
    # ローカルにデータセットが既に存在すればそれを再利用させる
    wiki_ja = load_dataset(
        "singletongue/wikipedia-utils",
        "passages-c400-jawiki-20240401",
        split="train",
        cache_dir="./datasets",
        trust_remote_code=True,
    )
    logger.info(f"{wiki_ja=}")
    return wiki_ja


def main():
    wiki_ja = load_wikipedia_japanese_datasets()
    # 96個ずつのバッチに分割
    batches = [wiki_ja[i : i + BATCH_SIZE] for i in range(0, len(wiki_ja), BATCH_SIZE)]
    print(f"{len(batches)=}")
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(batch_processing, batch): batch for batch in batches}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"バッチ処理中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
