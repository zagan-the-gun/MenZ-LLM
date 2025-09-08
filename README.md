# MenZ-LLM Realtime WS Client

zagaroid に WebSocket クライアントとして接続し、字幕を受信して LLM コメントを返すクライアント実装です。

方針:
- venv は手動で作成・有効化
- 設定は `config.ini` のみを使用（コマンドラインオプションは使わない）

参考:
- リアルタイム字幕: `MenZ-ReazonSpeech` (`https://github.com/zagan-the-gun/MenZ-ReazonSpeech`)
- zagaroid: `https://github.com/zagan-the-gun/zagaroid`

## セットアップ

仮想環境は手動で用意し、有効化します。

```bash
python -m venv .venv
source .venv/bin/activate
./setup.sh
```

`config.ini` を編集

```ini
[client]
host = localhost
port = 50001
path = /wipe_subtitle
reconnect_initial_ms = 500
reconnect_max_ms = 5000

[llm]
# モデル名と量子化を指定（自動解決）
model_name =
quantization = Q4_K_M
n_ctx = 2048
n_threads = 0   # 0は自動
n_gpu_layers = 0
```

## 起動

仮想環境を有効化した状態で実行します。

```bash
source .venv/bin/activate
./run.sh
```

## WebSocket プロトコル（初期版）

- 接続先: `ws://{host}:{port}{path}`（例: `ws://localhost:50001/wipe_subtitle`）
- 受信: 字幕
```json
{"type":"subtitle","text":"今日は良い天気","speaker":"viewer"}
```
- 送信: コメント
```json
{"type":"comment","comment":"いいね！"}
```

## Unity(zagaroid) 接続メモ
- クライアントが `subtitle` を受信したら LLM に投げて `comment` を同一WSへ返却

## モデル
- ローカルLLM（推奨）: `[llm]` に `model_path` を指定（HTTP不要）