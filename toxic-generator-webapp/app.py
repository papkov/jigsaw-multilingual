import argparse

from flask import Flask, request, render_template
from flask_cors import CORS
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM

parser = argparse.ArgumentParser()
parser.add_argument("--port", default="8080")
parser.add_argument("--ip", default="127.0.0.1")
args = parser.parse_args()

tokenizer = XLMRobertaTokenizer.from_pretrained('novinsh/xlm-roberta-large-toxicomments-12k')
model = XLMRobertaForMaskedLM.from_pretrained('novinsh/xlm-roberta-large-toxicomments-12k', output_attentions=False)
model.eval()

def duplicates(lst, item):
	return [i for i, x in enumerate(lst) if x == item]


app = Flask(__name__, template_folder='.')
CORS(app)


@app.route("/")
def home_page():
	return render_template('index.html', host=args.ip, port=args.port) #"Hello World!"


@app.route('/autocomplete', methods=['POST'])
def predict():
	sentence = ""
	sentence_orig = request.form.get('text')
	sentence_length = request.form.get('len')
	decoding_type = "" #request.form.get('decoding_type') # TODO
	domain_type = "" # request.form.get('domain_type') # TODO

	filler = ' '.join([tokenizer.mask_token for _ in range(int(sentence_length))])

	if domain_type == 'toxic':
		starter = '[TOXIC]'
	else:
		starter = ''

	if len(sentence_orig.strip()) == 0:
		sentence = f"{tokenizer.cls_token} {starter} {filler} . {tokenizer.sep_token}"
	else:
		sentence = f"{tokenizer.cls_token} {starter} {sentence_orig} {filler} . {tokenizer.sep_token}"

	print(sentence)

	tokenized_text = tokenizer.tokenize(sentence)

	##### LOOP TO CREATE TEXT #####
	generated = 0
	full_sentence = []
	while generated < int(sentence_length):
		mask_idxs = duplicates(tokenized_text, tokenizer.mask_token)

		if decoding_type == 'left to right':
			focus_mask_idx = min(mask_idxs)
		else:
			focus_mask_idx = np.random.choice(mask_idxs)

		mask_idxs.pop(mask_idxs.index(focus_mask_idx))
		temp_tokenized_text = tokenized_text.copy()
		temp_tokenized_text = [j for i, j in enumerate(temp_tokenized_text) if i not in mask_idxs]
		temp_indexed_tokens = tokenizer.convert_tokens_to_ids(temp_tokenized_text)
		ff = [idx for idx, i in enumerate(temp_indexed_tokens) if i == tokenizer.mask_token_id]
		temp_segments_ids = [0] * len(temp_tokenized_text)
		tokens_tensor = torch.tensor([temp_indexed_tokens])
		segments_tensors = torch.tensor([temp_segments_ids])

		with torch.no_grad():
			outputs = model(tokens_tensor, token_type_ids=segments_tensors)
			predictions = outputs[0]

		#     print(ff)
		#     print(predictions[0, ff])
		# TOP - k Sampling
		k = 5
		predicted_index = np.random.choice(predictions[0, ff].argsort()[0][-k:]).item()
		predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
		tokenized_text[focus_mask_idx] = predicted_token
		generated += 1

	generated_text = ' '.join(tokenized_text[1:-1]).replace('[ TOXIC ]', '')
	generated_text = ''.join([t for t in generated_text if t != '▁'])
	return generated_text


if __name__=='__main__':
	print(int(args.port))
	print(args.ip)
	app.run(debug=False, host=args.ip, port=int(args.port))

