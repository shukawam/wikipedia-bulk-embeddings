# Bulk embeddings from Wikipedia(JA) datasets

Wikipedia(日本語)データセットの Embedding を取得するためのスクリプトです。

## preparation

必要なライブラリをダウンロードします。

```sh
pip install -r requirements.txt
```

以下のように環境変数を設定します。

```sh
export COMPARTMENT_ID="<your-compartment-id>"
# Optional
export REGION="us-chicago-1"
export USE_IP="True"
```

また、当スクリプトは OCI Generative AI Service が使用可能なリージョンがサブスクライブ済みであり、それを使うための認可情報（Config or Instance Principal）が設定済みであることを想定しています。

## execute

```sh
nohup python bulk-embeddings.py
```
