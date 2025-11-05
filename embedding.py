import torch
from esm import pretrained

device = torch.device("cpu")
esm_model, alphabet = pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
esm_model.eval()
batch_converter = alphabet.get_batch_converter()

def get_esm_embedding(sequence):
    sequence = sequence.strip().replace(" ", "").upper()
    data = [("protein", sequence[:1022])]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        result = esm_model(tokens, repr_layers=[6], return_contacts=False)
        rep = result["representations"][6]
        emb = rep[0, 1:len(sequence)+1].mean(0).cpu().unsqueeze(0)
    return emb
