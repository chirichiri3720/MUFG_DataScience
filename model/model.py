import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig

from transformers import AutoModelForSequenceClassification, AutoConfig



class BertModel(nn.Module):
    def __init__(self, device, num_labels=6, model_config=None):
        super(BertModel, self).__init__()
        self.device = device
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.model = self.model.to(self.device)
        config = BertConfig(**model_config)
        self.model.config = config

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def add_layer(self,additional_layers=1):
        hidden_size = self.model.config.hidden_size
        in_features = self.model.classifier.in_features
        out_features = 6

        # 新しい分類器の構築
        classifier_layers = []
        for _ in range(additional_layers):
            classifier_layers.append(nn.Linear(in_features, hidden_size))
            classifier_layers.append(nn.ReLU())
            in_features = hidden_size
        
        classifier_layers.append(nn.Linear(in_features, out_features))

        self.model.classifier = nn.Sequential(*classifier_layers)
        self.model.to(self.device)

class DebertaModel(nn.Module):
    def __init__(self, device, num_labels=6, model_config=None):
        super(DebertaModel, self).__init__()

        self.device = device
        config = AutoConfig.from_pretrained('microsoft/deberta-v3-small')
        config.num_labels = num_labels
        
        # モデル設定をカスタマイズ（必要に応じて）
        if model_config:
            for key, value in model_config.items():
                setattr(config, key, value)
        
        self.model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-small', config=config)
        self.model.to(self.device)
        self.classifier = self.model.classifier

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def resize_token_embeddings(self, new_vocab_size):
        self.model.resize_token_embeddings(new_vocab_size)
    
    def add_layer(self,additional_layers=1):
        # hidden_size = self.model.model.config.hidden_size if hasattr(self.model.model, 'config') else 768  # or any appropriate default
        hidden_size = self.model.config.hidden_size
        in_features = self.model.classifier.in_features
        out_features = self.model.config.num_labels

        # 新しい分類器の構築
        classifier_layers = []
        for _ in range(additional_layers):
            classifier_layers.append(nn.Linear(in_features, hidden_size))
            classifier_layers.append(nn.ReLU())
            in_features = hidden_size
        
        classifier_layers.append(nn.Linear(in_features, out_features))

        self.model.classifier = nn.Sequential(*classifier_layers)
        self.model.to(self.device)


class RobertaModel(nn.Module):
    def __init__(self, device,num_labels=6, model_config=None):
        super(RobertaModel, self).__init__()

        self.device = device
        config = AutoConfig.from_pretrained('roberta-base')
        config.num_labels = num_labels
        
        # モデル設定をカスタマイズ（必要に応じて）
        if model_config:
            for key, value in model_config.items():
                setattr(config, key, value)
        
        self.model = AutoModelForSequenceClassification.from_pretrained('roberta-base', config=config)
        self.model.to(self.device)
        self.classifier = self.model.classifier

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits
    
    def resize_token_embeddings(self, new_vocab_size):
        self.model.resize_token_embeddings(new_vocab_size)
    
    def add_layer(self,additional_layers=1):
        # hidden_size = self.model.model.config.hidden_size if hasattr(self.model.model, 'config') else 768  # or any appropriate default
        hidden_size = self.model.config.hidden_size
        out_features = 6
        # in_features = self.model.classifier.in_features
        if hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'dense'):
            in_features = self.model.classifier.dense.in_features
        else:
            raise AttributeError("The classifier does not have 'in_features' attribute")

        # 新しい分類器の構築
        classifier_layers = []
        for _ in range(additional_layers):
            classifier_layers.append(nn.Linear(in_features, hidden_size))
            classifier_layers.append(nn.ReLU())
            in_features = hidden_size
        
        classifier_layers.append(nn.Linear(in_features, out_features))

        self.model.classifier = nn.Sequential(*classifier_layers)
        self.model.to(self.device)

class DistilbertModel(nn.Module):
    def __init__(self, device, num_labels=6, model_config=None):
        super(DistilbertModel, self).__init__()

        self.device = device
        config = AutoConfig.from_pretrained('distilbert-base-uncased')
        config.num_labels = num_labels
        
        # モデル設定をカスタマイズ（必要に応じて）
        if model_config:
            for key, value in model_config.items():
                setattr(config, key, value)
        
        self.model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', config=config)
        self.model.to(self.device)
        self.classifier = self.model.classifier
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def add_layer(self, additional_layers=1):
        hidden_size = self.model.config.hidden_size
        if isinstance(self.model.classifier, nn.Linear):
            in_features = self.model.classifier.in_features
        else:
            raise AttributeError("The classifier does not have 'in_features' attribute")
        out_features = self.model.config.num_labels

        # 新しい分類器の構築
        classifier_layers = []
        for _ in range(additional_layers):
            classifier_layers.append(nn.Linear(in_features, hidden_size))
            classifier_layers.append(nn.ReLU())
            in_features = hidden_size
        
        classifier_layers.append(nn.Linear(in_features, out_features))

        self.model.classifier = nn.Sequential(*classifier_layers)
        self.model.to(self.device)

class XLNetModel(nn.Module): #add_layerを消したら動く
    def __init__(self, device, num_labels=6, model_config=None):
        super(XLNetModel, self).__init__()

        self.device = device
        config = AutoConfig.from_pretrained('xlnet-base-cased')
        config.num_labels = num_labels
        
        # モデル設定をカスタマイズ（必要に応じて）
        if model_config:
            for key, value in model_config.items():
                setattr(config, key, value)
        
        self.model = AutoModelForSequenceClassification.from_pretrained('xlnet-base-cased',num_labels=num_labels)
        self.model.to(self.device)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        return outputs
    
    def resize_token_embeddings(self, new_vocab_size):
        self.model.resize_token_embeddings(new_vocab_size)
    
    def add_layer(self, additional_layers=1):
        hidden_size = self.model.config.hidden_size
        
        # Classifierがnn.Linearかどうか確認する
        if hasattr(self.model, 'logits_proj') and isinstance(self.model.logits_proj, nn.Linear):
            in_features = self.model.logits_proj.in_features
        else:
            raise AttributeError("The classifier does not have 'in_features' attribute")
        
        out_features = self.model.config.num_labels

        # 新しい分類器の構築
        classifier_layers = []
        for _ in range(additional_layers):
            classifier_layers.append(nn.Linear(in_features, hidden_size))
            classifier_layers.append(nn.ReLU())
            in_features = hidden_size
        
        classifier_layers.append(nn.Linear(in_features, out_features))

        self.model.logits_proj = nn.Sequential(*classifier_layers)
        self.model.to(self.device)

class ALbertModel(nn.Module): #addd_layerを消したら動く
    def __init__(self, device, num_labels=6, model_config=None):
        super(ALbertModel, self).__init__()

        self.device = device
        config = AutoConfig.from_pretrained('albert-base-v2')
        config.num_labels = num_labels
        
        # モデル設定をカスタマイズ（必要に応じて）
        if model_config:
            for key, value in model_config.items():
                setattr(config, key, value)
        
        self.model = AutoModelForSequenceClassification.from_pretrained('albert-base-v2', config=config)
        self.model.to(self.device)
        self.classifier = self.model.classifier
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def resize_token_embeddings(self, new_vocab_size):
        self.model.resize_token_embeddings(new_vocab_size)
    
    def add_layer(self, additional_layers=1):
        hidden_size = self.model.config.hidden_size
        
        # Classifierがnn.Linearかどうか確認する
        if isinstance(self.model.classifier, nn.Linear):
            in_features = self.model.classifier.in_features
        else:
            raise AttributeError("The classifier does not have 'in_features' attribute")
        
        out_features = self.model.config.num_labels

        # 新しい分類器の構築
        classifier_layers = []
        for _ in range(additional_layers):
            classifier_layers.append(nn.Linear(in_features, hidden_size))
            classifier_layers.append(nn.ReLU())
            in_features = hidden_size
        
        classifier_layers.append(nn.Linear(in_features, out_features))

        self.model.classifier = nn.Sequential(*classifier_layers)
        self.model.to(self.device)
    
class ElectraModel(nn.Module):
    def __init__(self, device, num_labels=6, model_config=None):
        super(ElectraModel, self).__init__()

        self.device = device
        config = AutoConfig.from_pretrained('google/electra-base-discriminator')
        config.num_labels = num_labels
        
        # モデル設定をカスタマイズ（必要に応じて）
        if model_config:
            for key, value in model_config.items():
                setattr(config, key, value)
        
        self.model = AutoModelForSequenceClassification.from_pretrained('google/electra-base-discriminator', config=config)
        self.model.to(self.device)
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def resize_token_embeddings(self, new_vocab_size):
        self.model.resize_token_embeddings(new_vocab_size)
    
    def add_layer(self, additional_layers=1):
        hidden_size = self.model.config.hidden_size
        out_features = self.model.config.num_labels

        # 元の分類層を取得
        if hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'in_features'):
            in_features = self.model.classifier.in_features
        else:
            in_features = hidden_size  # デフォルト値を使用
        
        # 新しい分類器の構築
        classifier_layers = []
        for _ in range(additional_layers):
            classifier_layers.append(nn.Linear(in_features, hidden_size))
            classifier_layers.append(nn.ReLU())
            in_features = hidden_size
        
        classifier_layers.append(nn.Linear(in_features, out_features))

        self.model.classifier = nn.Sequential(*classifier_layers)
        self.model.to(self.device)