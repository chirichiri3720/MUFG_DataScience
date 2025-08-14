from transformers import BertTokenizer, BertForSequenceClassification,BertConfig

# モデル名
model_name = 'bert-base-uncased'

# トークナイザーのダウンロード
tokenizer = BertTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('./bert-base-uncased-tokenizer')


