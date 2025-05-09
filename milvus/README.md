# Milvus

`bulk_embeddings.py` で得られた大規模ベクトルデータセットを用いて、Milvus でベクトルインデックスの作成時間やベクトル検索にかかる時間を測定するためのスクリプトです。

## preparetion

`bulk_embeddings.py` を実行し、得られた CSV ファイルを`data/`へ格納します。

`.env.sample` を参考に `.env` を作成します。

## Scripts

各スクリプトは以下のような動作をします。

- `01_data_loader.py`: `data/<wikipedia-dataset>.csv` を読み込み、Milvus に格納します。
- `02_data+checker.py`: Collection が正しく作成されていることと、挿入したデータ数を数えます。
- `03_01_create_ivf_index.py`: `IVF_FLAT` のベクトルインデックスを作成し、その作成に要した時間とインデックスの詳細を出力します。べき等に作っているので、HNSW のインデックス等が作成済みの場合でもそのまま実行可能です。
- `03_02_create_hnsw_index.py`: `HNSW` のベクトルインデックスを作成し、その作成に要した時間とインデックスの詳細を出力します。べき等に作っているので、IVF_FLAT のインデックス等が作成済みの場合でもそのまま実行可能です。
- `04_load_collection.py`: Collection や作成した Index をメモリ上にロードします。`05_vector_search.py` の実行前に必ず実行してください。
- `05_vector_search.py`: ベクトル検索を行い、その実行時間と得られた結果を出力します。
