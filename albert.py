from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
import torch

# モデル名
model_name = 'albert-base-v2'

# モデルのダウンロード
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ターゲットの数を変更（分類器の再定義）
model.classifier = torch.nn.Linear(model.config.hidden_size, 6)

# 新しい分類器のドロップアウトプロパティをモデル設定に反映
model.config.num_labels = 6

model.save_pretrained('albert-base-v2-model')

tokenizer.save_pretrained('albert-base-v2-tokenizer')