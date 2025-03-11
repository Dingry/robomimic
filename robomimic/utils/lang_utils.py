import os
import torch
from transformers import AutoModel, pipeline, AutoTokenizer, CLIPTextModelWithProjection, AutoModelForTextEncoding

class LangEncoder:
    def __init__(self, device, model_variant="clip"):
        os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
        
        self.device = device
        self.model_variant = model_variant
        
        if self.model_variant == "clip":
            model_variant = "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32"
            self.lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
                model_variant,
                cache_dir=os.path.expanduser("~/tmp/clip")
            ).to(device).eval()
            self.tz = AutoTokenizer.from_pretrained(model_variant, TOKENIZERS_PARALLELISM=True)
        elif self.model_variant == "t5":
            self.lang_emb_model = AutoModelForTextEncoding.from_pretrained("t5-small").to(device).eval()
            self.tz = AutoTokenizer.from_pretrained("t5-small")
        else:
            raise ValueError("Invalid language encoder model")
        
        self.lang_emb_model.eval()

    def get_lang_emb(self, lang):
        if lang is None:
            return None
 
        with torch.no_grad():
            tokens = self.tz(
                text=lang,                   # the sentence to be encoded
                add_special_tokens=True,             # Add [CLS] and [SEP]
                # max_length=25,  # maximum length of a sentence
                padding="max_length",
                return_attention_mask=True,        # Generate the attention mask
                return_tensors="pt",               # ask the function to return PyTorch tensors
            ).to(self.device)

            if self.model_variant == "clip":
                lang_emb = self.lang_emb_model(**tokens)['text_embeds'].detach()
                # print(lang_emb.shape)
                # exit()
            elif self.model_variant == "t5":
                lang_emb = self.lang_emb_model(**tokens)['last_hidden_state'].detach()
                lang_emb = torch.mean(lang_emb, dim=1)
                # print(lang_emb.shape)
                # exit()
        
        # check if input is batched or single string
        if isinstance(lang, str):
            lang_emb = lang_emb[0]

        return lang_emb

