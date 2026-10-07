# OmniDictate-CLI

Windows用リアルタイムAI音声認識ツール。faster-whisperを使用し、どこでも瞬時に音声をテキストに変換します。

## 目次

- [機能](#機能)
- [システム要件](#システム要件)
- [インストール](#インストール)
- [使用方法](#使用方法)
- [設定](#設定)
- [技術仕様](#技術仕様)
- [アーキテクチャ](#アーキテクチャ)
- [開発者ガイド](#開発者ガイド)
- [トラブルシューティング](#トラブルシューティング)
- [最適化履歴](#最適化履歴)
- [ライセンス](#ライセンス)

## 機能

- **リアルタイム転写**: 音声をリアルタイムでテキストに変換
- **トグル式録音**: 右Shiftキーで録音ON/OFFを切り替え
- **音声活動検出（VAD）**: 録音ON時、音声を検出すると自動的に転写開始
- **Bluetooth音質維持**: マイクはON時のみ起動（HFP/HSP切り替えを最小化）
- **音声コマンド**: 単語削除、改行挿入、句読点入力
- **幻覚軽減**: 反復的・無意味なテキストの自動フィルタリング
- **GPU高速化**: CUDA対応でリアルタイム転写を実現

## システム要件

- **OS**: Windows 10/11
- **Python**: 3.13（動作確認は 3.13.5。`requirements.txt` の版は 3.11 以上が前提）
- **RAM**: 8GB以上（16GB推奨）
- **GPU**: CUDA対応NVIDIA GPU（6GB以上VRAM推奨）
  - 動作確認済み: RTX 4070 Ti (12GB VRAM)
- **CUDA**: CUDAツールキット + cuDNN

## インストール

1. リポジトリをクローン
2. 仮想環境を作成・有効化:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. 依存関係をインストール:
   ```bash
   pip install -r requirements.txt
   ```
   PyTorch の CUDA 版（CUDA 12.4 用）もここで一緒に入る。動作中の環境の版で固定してある。
4. CUDA 12.4 以外を使う場合は、`requirements.txt` の `cu124` と `torch` の版を書き換えてから実行する（[公式サイト](https://pytorch.org/get-started/locally/)参照）

## 使用方法

### 起動

**日本語版**: `start.bat` をダブルクリック
**英語版**: `start_en.bat` をダブルクリック

### 基本操作

| キー | 動作 |
|------|------|
| **右Shift** | 録音ON/OFF切り替え |
| **Esc** | アプリ終了 |

### 動作フロー

1. アプリ起動 → マイクOFF（Bluetooth高音質維持）
2. 右Shift押下 → マイクON（録音開始、ビープ音：高音）
3. 音声検出 → 自動転写 → アクティブウィンドウに入力
4. 右Shift押下 → マイクOFF（録音停止、ビープ音：低音）

### 音声コマンド

**テキスト操作**:
- `delete last [n] words` - 最後のn単語を削除
- `new line` / `next line` - 改行挿入

**句読点**（英語音声で入力）:
- `period` / `comma` / `question mark` / `exclamation mark`
- `colon` / `semicolon` / `hyphen` / `dash`
- `open/close parenthesis` / `open/close bracket` / `open/close brace`

### 直送モード（フォーカスに関係なく送る）

通常は書き起こしをクリップボード経由で**前面のウィンドウ**へ貼り付けます。そのため別のアプリをクリックすると、言葉がそちらへ入ってしまいます。
直送モードでは貼り付けの代わりに、決まった相手の入力欄へ直接届けます（日本語版のみ）。

| キー | 送り先 | 届け方 |
|------|------|------|
| **Ctrl+Alt+F9** | Irodori（音声チャット）の入力欄 | Irodori のバックエンド `POST /api/dictation` へ送り、UI が入力欄へ入れる |
| **Ctrl+Alt+F10** | local-claude（ターミナルの Claude Code）の入力欄 | そのコンソールへキー入力として直接書き込む（`console_bridge.py`）。送り先は local-claude が起動時に書く `state\console.json` |

- 同じキーでもう一度押すと OFF。F9 と F10 は同時に ON になりません。音で知らせます（Irodori ON: 高い音2回／local-claude ON: 上がる音3回／OFF: 低い音1回）。
- 起動時の状態はランチャーの「直送先」（なし / Irodori / local-claude）で選べます（`dictation.py --direct irodori|lc`）。
- **送信**: 文末の「どうぞ。」「どうぞ」、音声の「エンター」で送信します。送信できたら「ピロン」（880→1320Hz の2音。Irodori の UI の送信チャイムと同じ音）が鳴ります。Irodori への「どうぞ」は Irodori の UI が判定して鳴らします（二重に鳴らないよう、OmniDictate では鳴らしません）。
- **local-claude の確認画面**: ツールの許可や選択肢が出ているときに Enter を送ると、既定の選択（「Yes」など）で決まってしまいます。
  そこで Enter の前に画面を読み、確認・選択の画面なら**1回目は送らずに警告音（下がる音2回）だけ**鳴らします。
  警告から20秒以内にもう一度「どうぞ」「エンター」と言えば送ります。
- **届かないとき**（相手が起動していない等）は、貼り付けずにクリップボードへ残し、低い音2回で知らせます。言葉は失われず、作業中のアプリにも入りません。
- 直送中は `delete last N words` は使えません（前面のウィンドウを操作する仕組みのため）。

## 設定

`dictation.py` または `dictation_en.py` で変更可能：

### 音声感度

```python
SENSITIVITY_LEVEL = 2  # 1:低, 2:中, 3:高, 4:超高
```

| レベル | 用途 |
|--------|------|
| 1 | 雑音が多い環境 |
| 2 | 通常（デフォルト） |
| 3 | 小声での入力 |
| 4 | 囁き声対応 |

### Whisperモデル

```python
MODEL_SIZE = "large-v3"  # tiny/base/small/medium/large-v3
```

小さいモデルほど高速だが精度は低下。

### キー設定

```python
RECORD_KEY = keyboard.Key.shift_r  # 録音トグルキー
STOP_KEY = keyboard.Key.esc        # 終了キー
```

### 言語変更

`model.transcribe()`内の`language`パラメータを変更：
- 日本語: `language="ja"`
- 英語: `language="en"`
- その他: Whisper対応の2文字言語コード

## 技術仕様

### アーキテクチャ概要

```
[マイク入力] → [音声増幅] → [VAD検出] → [バッファリング] → [Whisper転写] → [フィルタリング] → [キーボード入力]
     ↓              ↓            ↓              ↓                 ↓                ↓                  ↓
  sounddevice  プリエンファシス webrtcvad   スレッド処理    faster-whisper    幻覚除去         pywinauto
```

### コアコンポーネント

#### 1. 音声入力システム
- **ライブラリ**: sounddevice
- **サンプリングレート**: 16000Hz (Whisper最適値)
- **チャンネル**: モノラル
- **データ型**: int16 → float32変換
- **処理**:
  - 増幅フィルター（最大値正規化）
  - プリエンファシスフィルター（高周波強調、係数0.97）

#### 2. 音声活動検出（VAD）
- **ライブラリ**: webrtcvad
- **モード設定**: SENSITIVITY_LEVEL に基づく (0-3)
- **フレームサイズ**: 30ms (480サンプル @ 16kHz)
- **検出ロジック**:
  - 音声フレーム連続3回で録音開始
  - 無音フレーム連続30回（約900ms）で録音停止

#### 3. Whisper音声認識
- **モデル**: faster-whisper (CTranslate2最適化版)
- **サイズ**: large-v3 (デフォルト)
- **コンピュート**: float16 (GPU高速化)
- **最適化パラメータ**:
  ```python
  beam_size=5              # ビームサーチ幅（速度と精度のバランス）
  temperature=0.0          # 決定的出力（ランダム性なし）
  condition_on_previous_text=False  # 文脈ドリフト防止
  no_speech_threshold=0.3  # 音声検出閾値（低めで感度向上）
  ```

#### 4. 幻覚フィルタリング
- **対象**: YouTube終了フレーズなど
- **方式**: 完全一致フィルタリング（部分一致は使用しない）
- **日本語例**: "ご視聴ありがとうございました"
- **英語例**: "thanks for watching", "like and subscribe"

#### 5. テキスト入力
- **ライブラリ**: pywinauto (Win32 API)
- **方式**: Ctrl+V によるクリップボード経由の貼り付け
- **遅延**: 0.1秒（アプリ応答待ち）

## アーキテクチャ

### スレッド構成

| スレッド | 役割 | 処理内容 |
|----------|------|----------|
| **メインスレッド** | UI制御 | キーボードリスナー、アプリ終了制御 |
| **音声入力スレッド** | 録音 | sounddeviceコールバック、バッファ追加 |
| **転写スレッド** | 音声認識 | Whisper実行、テキスト出力 |

### データフロー

1. **音声入力**: `audio_callback()` → `audio_queue`
2. **VAD検出**: `vad.is_speech()` → 音声/無音判定
3. **バッファ蓄積**: `audio_buffer` (deque) → 最大50チャンク
4. **転写トリガー**: 無音検出 → `transcription_event.set()`
5. **Whisper実行**: `transcription_thread()` → GPU処理
6. **テキスト出力**: `type_text()` → クリップボード経由

### メモリ管理

- **音声バッファ**: deque (最大50チャンク = 約15秒)
- **GPU VRAM**: モデルサイズ依存
  - large-v3: 約5GB
  - medium: 約2GB
  - small: 約1GB

## 開発者ガイド

### ファイル構成

```
OmniDictate-CLI/
├── dictation.py          # 日本語版メインプログラム
├── dictation_en.py       # 英語版メインプログラム
├── console_bridge.py     # 別のコンソールへのキー入力の書き込み・画面の読み取り（local-claude 直送）
├── launcher.py           # 設定つきの起動画面（OmniDictate.bat から起動）
├── start.bat             # 日本語版ランチャー
├── start_en.bat          # 英語版ランチャー
├── requirements.txt      # Python依存関係
├── README.md             # このファイル
└── venv/                 # 仮想環境 (自動生成)
```

### 主要な関数

#### `dictation.py` / `dictation_en.py`

| 関数名 | 行番号 | 役割 |
|--------|--------|------|
| `load_model()` | ~100 | Whisperモデル読み込み、GPU検出 |
| `audio_callback()` | ~200 | 音声入力コールバック、VAD実行 |
| `transcription_thread()` | ~480 | 転写スレッド、Whisper実行 |
| `type_text()` | ~560 | テキスト入力、クリップボード操作 |
| `on_press()` / `on_release()` | ~680 | キーボードイベント処理 |

### デバッグログの見方

アプリケーションは詳細なログを出力します：

```
[AUDIO] Audio level: 245.6
→ 音声レベル（0-32768）。100以上で音声あり

[VAD] Voice detected: True, Consecutive frames: 3
→ VAD検出結果。Consecutive frames: 3で録音開始

[TRANSCRIPTION] *** STARTING TRANSCRIPTION ***
[TRANSCRIPTION] Audio samples: 28800, Chunks: 18, Duration: 1.80s
→ 転写開始。サンプル数、チャンク数、音声時間

[TRANSCRIPTION] *** TRANSCRIPTION COMPLETED ***
[TRANSCRIPTION] Language: ja, Probability: 0.95
[TRANSCRIPTION] Duration: 1.80s
[TRANSCRIPTION] Segments: 1
→ 転写完了。言語確率、セグメント数

[TRANSCRIPTION] Full text: 'こんにちは'
→ 認識されたテキスト

[TYPING] Typing text: 'こんにちは'
→ 入力されるテキスト
```

### パラメータチューニングガイド

#### 問題：音声が認識されない

**症状**: "Segments: 0, Full text: ''" が頻発

**確認ポイント**:
1. 音声レベル（100以上必要）
2. VAD検出（Voice detected: True）
3. 音声時間（1秒以上推奨）

**調整方法**:
```python
# 感度を上げる
SENSITIVITY_LEVEL = 3  # または 4

# Whisper閾値を下げる
no_speech_threshold=0.2  # デフォルト: 0.3
```

#### 問題：幻覚出力が多い

**症状**: "ご視聴ありがとうございました" など無関係なテキスト

**調整方法**:
```python
# 幻覚フィルターに追加 (dictation.py:537行目付近)
hallucination_phrases = (
    "あなたの幻覚フレーズ",
    # ... 既存のフレーズ
)

# または no_speech_threshold を上げる
no_speech_threshold=0.4  # より厳格
```

#### 問題：転写が遅い

**調整方法**:
```python
# モデルサイズを下げる
MODEL_SIZE = "medium"  # または "small"

# ビームサイズを下げる
beam_size=3  # デフォルト: 5
```

### コード変更時の注意点

#### 1. Whisperパラメータの変更

**重要**: 以下のパラメータは設定しないこと:
- `vad_filter=True` - アプリ独自のVADと競合し、二重フィルタリングで音声が遮断される
- `initial_prompt` - バイアスがかかり幻覚の原因になる

**推奨設定**:
```python
segments, info = model.transcribe(
    audio_float32,
    beam_size=5,
    language="ja",  # または "en"
    temperature=0.0,
    condition_on_previous_text=False,
    no_speech_threshold=0.3
)
```

#### 2. VADパラメータの変更

```python
# 音声検出の連続フレーム数（少なすぎると誤検出）
voice_frames_threshold = 3  # 推奨: 3-5

# 無音検出の連続フレーム数（多すぎると反応が遅い）
silence_frames_threshold = 30  # 推奨: 20-40
```

#### 3. エンコーディング

**必須**: すべてのPythonファイルに以下を追加:
```python
# -*- coding: utf-8 -*-
```

バッチファイルに以下を追加:
```batch
set PYTHONUTF8=1
```

これにより日本語Windowsでの文字化けを防止。

#### 4. バッチファイルのパス

**必須**: 相対パス使用:
```batch
cd /d "%~dp0"
venv\Scripts\python.exe dictation.py
```

絶対パスは使用しない（ポータビリティのため）。

## Bluetooth使用時の注意

本アプリはBluetoothイヤホン使用時の音質劣化問題に対応しています：

- **起動時**: マイク未起動 → A2DP（高音質）維持
- **録音ON**: マイク起動 → HFP/HSP（通話品質）に切り替わる
- **録音OFF**: マイク停止 → A2DP（高音質）に戻る

録音ON時に100-300msの遅延が発生する場合があります。

## トラブルシューティング

| 問題 | 解決策 |
|------|--------|
| **CUDA not available** | PyTorch CUDA版を再インストール、CUDAツールキット/cuDNNを確認 |
| **音声入力なし** | マイク設定確認、16000Hz推奨 |
| **転写が遅い** | GPUを使用、または小さいモデルに変更 |
| **ModuleNotFoundError** | 仮想環境を有効化して `pip install -r requirements.txt` |
| **文字化け** | ファイル先頭に `# -*- coding: utf-8 -*-`、batch に `set PYTHONUTF8=1` |
| **ダブルクリックで起動しない** | batch ファイルのパスが相対パスになっているか確認 |
| **Segments: 0 が頻発** | SENSITIVITY_LEVEL を上げる、音声を大きくする |
| **幻覚が出る** | hallucination_phrases に追加、no_speech_threshold を調整 |

### デバッグモード

詳細ログを確認するには、コマンドラインから実行:
```bash
cd C:\Apps\OmniDictate-CLI
venv\Scripts\activate
python dictation.py  # または dictation_en.py
```

## 最適化履歴

### 2025-12-10: VADフィルター問題修正

**問題**: Whisperの `vad_filter=True` により二重フィルタリングが発生し、正常な音声が遮断される

**修正内容**:
- `vad_filter=True` を削除
- `vad_parameters` を削除
- アプリ独自のwebrtcvad VADのみを使用

**結果**: テキスト出力が正常に機能

### 2025-12-10: 幻覚フィルター最適化

**問題**: 部分一致フィルターが正常なテキストまで遮断

**修正内容**:
- 部分一致フィルタリングを削除
- 完全一致のみに変更

**結果**: 誤検出が減少

### 2025-12-10: Whisperパラメータ最適化

**変更内容**:
- `beam_size`: 8 → 5（速度向上）
- `no_speech_threshold`: 0.6 → 0.3（感度向上）
- `initial_prompt` を削除（バイアス除去）
- `condition_on_previous_text`: False（文脈ドリフト防止）

**結果**: 認識精度と速度のバランス改善

### 2025-12-10: エンコーディング問題修正

**問題**: 日本語Windowsで起動時に文字化け

**修正内容**:
- Python ファイルに `# -*- coding: utf-8 -*-` を追加
- batch ファイルに `set PYTHONUTF8=1` を追加

**結果**: 文字化け解消

### 2025-12-10: ポータビリティ改善

**問題**: 絶対パスによりディレクトリ移動不可

**修正内容**:
- batch ファイルを相対パスに変更（`%~dp0` 使用）
- `call venv\Scripts\activate.bat` → `venv\Scripts\python.exe` に変更

**結果**: どのディレクトリでも動作可能に

## ライセンス

MIT License - Copyright (c) 2023 GURJAR1
