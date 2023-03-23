import torch
from pytorch_transformers import RobertaModel, RobertaTokenizer
from pytorch_transformers import RobertaForSequenceClassification, RobertaConfig

# Loading RoBERTa classes
config = RobertaConfig.from_pretrained('roberta-base')
config.num_labels = 4
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.cuda()

model_path = 'testing_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))

# Feature Preparation
def prepare_features(seq_1, max_seq_length = 300, 
             zero_pad = False, include_CLS_token = True, include_SEP_token = True):
    ## Tokenzine Input
    tokens_a = tokenizer.tokenize(seq_1)

    ## Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    ## Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    ## Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    ## Input Mask 
    input_mask = [1] * len(input_ids)
    ## Zero-pad sequence lenght
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), input_mask

def get_reply(msg):
  model.eval()
  input_msg, _ = prepare_features(msg)
  if torch.cuda.is_available():
    input_msg = input_msg.cuda()
  output = model(input_msg)[0]
  _, pred_label = torch.max(output.data, 1)
  prediction=list(label_to_ix.keys())[pred_label]
  return prediction

label_to_ix = {'drink spillage': 0,
                'wrong items': 1,
                'quality issues': 2,
                'plain statement': 3}

label_to_ix.keys()

pred = get_reply("tell your staff ya, next time carry the package properly, else it spills everywhere wei")
print(pred)