from model import BertModel,RobertaModel,DebertaModel,DistilbertModel,XLNetModel,ElectraModel

def get_optimizer_grouped_parameters(model,model_name, lr=1e-3, weight_decay=0.01, lr_decay=0.95):
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    if model_name == 'deberta':
        model_part = model.model.deberta
        layers = [model_part.embeddings] + list(model_part.encoder.layer)
    elif model_name == 'bert':
        model_part = model.model.bert
        layers = [model_part.embeddings] + list(model_part.encoder.layer)
    elif model_name == 'roberta': #add_layer消す なし
        model_part = model.model.roberta
        layers = [model_part.embeddings] + list(model_part.encoder.layer)
    elif model_name == 'distilbert':
        model_part = model.model.distilbert
        layers = [model_part.embeddings] + list(model_part.transformer.layer)
    elif model_name == 'xlnet':
        model_part = model.model.transformer
        layers = list(model_part.layer)
    elif model_name == 'albert':
        model_part = model.model.albert
        layers = [model_part.embeddings] + list(model_part.encoder.albert_layer_groups)
    elif model_name == 'electra':
        model_part = model.model.electra
        layers = [model_part.embeddings] + list(model_part.encoder.layer)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    layers.reverse()
    lr_layer = lr#層ごとの学習率の初期値

    for layer in layers:
        lr_layer *= lr_decay 
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr_layer,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr_layer,
            },
        ]
    
    optimizer_grouped_parameters += [
        {
            "params": [p for n, p in model.named_parameters() if 'embeddings' not in n and 'encoder' not in n and 'transformer' not in n],
            "weight_decay": weight_decay,
            "lr": lr,
        },
    ]

    
    return optimizer_grouped_parameters
