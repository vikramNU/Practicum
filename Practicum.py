import torch
from transformers import AutoModelForCausalLM, \
  AutoTokenizer
import numpy as np
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, OPTModel

prompt = "The child was playing with a"
print("\nInput sequence: ")
print(prompt)

"""#Paraphasing"""

model_name = 'tuner007/pegasus_paraphrase' 
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
PegasusTokenizer = PegasusTokenizer.from_pretrained(model_name)
PegasusModel = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

tgt_text=[]
num_beams = 10              # bigger = more accurate, but longer search time
num_return_sequences = 5
batch = PegasusTokenizer([prompt],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
translated = PegasusModel.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
tgt_text=PegasusTokenizer.batch_decode(translated, skip_special_tokens=True)
tgt_text.append(prompt)
print(tgt_text)

"""#GPT-2 Transformer"""

GPTTokenizer = AutoTokenizer.from_pretrained("gpt2")
GPTModel = AutoModelForCausalLM.from_pretrained("gpt2")

wordprob=[]
for x in tgt_text:
  inpts = GPTTokenizer(x[:-1], return_tensors="pt")
  print("\nTokenized input data structure: ")
  print(inpts)

  inpt_ids = inpts["input_ids"]  # just IDS, no attn mask
  print("\nToken IDs and their words: ")
  for id in inpt_ids[0]:
    word = GPTTokenizer.decode(id)
    print(id, word)

  with torch.no_grad():
    logits = GPTModel(**inpts).logits[:, -1, :]
  print("\nAll logits for next word: ")
  print(logits)

  print("Probabilities",torch.softmax(logits, dim=-1))

  pred_id = torch.argmax(logits).item()
  print("\nPredicted token ID of next word: ")
  print(pred_id)

  pred_word = GPTTokenizer.decode(pred_id)
  print("\nPredicted next word for sequence: ")
  print(x[:-1],pred_word,'\n-------')
  wordprob.append([x,pred_word,"{:.6f}".format(torch.softmax(logits, dim=-1)[0][0].detach().numpy().tolist())])

"""#The child was playing with a - 1.4815e-06. The child played with a - 7.7249e-07"""

wordprob

