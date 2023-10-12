"""
Embedder Class

Example Usage:

doc = 'hello there! What\'s up? The market.'
embedder = Embedder()
embedded = embedder.get_sentence_embeddings(doc)
print(embedded.size())

>torch.Size([3, 768])

"""
from typing import Any

import gensim.downloader
import nltk
import torch
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from models.zsl import transformer_map
from nltk.sem.evaluate import Model
from nltk.tokenize import sent_tokenize
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

nltk.download("punkt")

device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedder:
    def __init__(self, transformer=None, pretrained_path=None, device=device_) -> None:
        self.w2v: Any = None  # word2vec
        self.w2v_size: int = -1
        self.gw: Any = None  # glove-wiki-gigaword-100
        self.transformer = transformer
        self.device = device

        self.transformers: dict[str, dict] = {}
        for k in transformer_map.keys():
            self.transformers[k] = {}
            self.transformers[k]["tokenizer"] = None
            self.transformers[k]["model"] = None
            if self.transformer is not None and k == self.transformer:
                if pretrained_path is not None:
                    self.transformers[k]["model"] = torch.load(pretrained_path)
                else:
                    self.transformers[k]["model"] = AutoModel.from_pretrained(
                        transformer_map[transformer]
                    )
                self.transformers[k]["model"] = self.transformers[k]["model"].to(device)
                self.transformers[transformer][
                    "tokenizer"
                ] = AutoTokenizer.from_pretrained(transformer_map[transformer])

    def _init(self, transformer):
        if transformer is None:
            transformer = self.transformer

        if transformer is None:
            raise ValueError(
                f"Must specifiy arg 'transformer' in one of {list(self.transformers.keys())}"
            )
        if transformer not in self.transformers.keys():
            raise ValueError(
                f"'transformer' argument must be one of {list(self.transformers.keys())}"
            )
        if self.transformers[transformer]["tokenizer"] is None:
            self.transformers[transformer]["tokenizer"] = AutoTokenizer.from_pretrained(
                transformer_map[transformer]
            )

        if self.transformers[transformer]["model"] is None:
            self.transformers[transformer]["model"] = AutoModel.from_pretrained(
                transformer_map[transformer]
            )

            for _, param in self.transformers[transformer]["model"].named_parameters():
                param.requires_grad = False

        return transformer

    def get_tokens(
        self,
        raw,
        transformer: str = None,
        # max_sentence_length: int = 256,
        # ) -> tuple[torch.Tensor, torch.Tensor]:
    ) -> Any:
        transformer = self._init(transformer)
        tokenizer = self.transformers[transformer]["tokenizer"]
        # print(
        #     "decoded: ",
        #     tokenizer.decode(
        #         tokenizer(raw, return_tensors="pt")["input_ids"]
        #         .reshape((-1,))
        #         .squeeze()
        #     ),
        # )
        return (
            tokenizer(raw, return_tensors="pt")["input_ids"]
            .reshape((-1,))[1:-1]
            .squeeze()
        )

        # doc_lengths = []
        # sentences = []
        # for doc in docs:
        #     for sent in doc:

        #     doc_lengths.append(num_sents)

        # tokenizer = self.transformers[transformer]["tokenizer"]
        # tokens = tokenizer(sentences, return_tensors="pt", padding="longest")[
        #     "input_ids"
        # ]

        # tokenized = []
        # for length in doc_lengths:
        #     tokenized.append(tokens[:length])
        #     tokens = tokens[length:]

        # padded = pad_sequence(tokenized, batch_first=True, padding_value=0.0)

        # return padded.to(device), torch.tensor(doc_lengths).to(device)

    def get_embedding(self, raw: str, transformer: str = None):
        """Returns embedding and tokens of raw string

        Args:
            raw (str):
            transformer (str, optional): Which transformer key to use. Defaults to initialized transformer.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: size(num tokens x 768) embedding, size(num tokens)
        """
        transformer = self._init(transformer)

        model = self.transformers[transformer]["model"]
        tokenizer = self.transformers[transformer]["tokenizer"]
        encoded = tokenizer(raw, return_tensors="pt")["input_ids"].to(self.device)
        output = model(encoded).last_hidden_state.squeeze()

        # print("")
        # print(tokenizer.decode(encoded["input_ids"].reshape((-1,)).tolist()))
        # print(tokenizer.decode(encoded["input_ids"].reshape((-1,))[1:-1].tolist()))
        # print("")
        return output[1:-1, :], encoded.reshape((-1,))[1:-1]

    def decode(self, tokens, transformer=None):
        transformer = self._init(transformer)
        tokenizer = self.transformers[transformer]["tokenizer"]
        return tokenizer.decode(tokens)

    def get_sentence_embedding(
        self,
        sentence: str,
        transformer: str = None,
        pooling: str = "mean",
        pretrained_model=None,
    ) -> torch.Tensor:
        """Generates sequence of sentence embeddings embedding for any raw string

        Args:
            raw (str): raw text
            transformer (str): transformer to use out of (bert, sbert_mean, sbert_cls, xlnet)
        Returns:
            torch.Tensor: Tensor of size (num sentences, embedding dimension)
        """
        transformer = self._init(transformer)

        tokenizer = self.transformers[transformer]["tokenizer"]
        model = self.transformers[transformer]["model"]

        if pretrained_model is not None:
            model = pretrained_model

        encoded = tokenizer(sentence, return_tensors="pt")

        with torch.no_grad():
            output = model(**encoded).last_hidden_state

        if pooling == "mean":
            # Mean Pooling
            input_mask_expanded = (
                encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
            )
            sum_embeddings = torch.sum(output * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return (sum_embeddings / sum_mask).squeeze()
        else:
            return output[:, 0, :].squeeze()

    def get_bert(self, raw: str) -> torch.Tensor:
        """Generates BERT embedding for any raw string

        Args:
            raw (str):

        Returns:
            torch.Tensor: Tensor of size (num words, 768)
        """
        self._init("bert")

        encoded = self.transformers["bert"]["tokenizer"](
            raw, add_special_tokens=True, return_tensors="pt"
        )
        encoded_ids = encoded["input_ids"]
        output = self.transformers["bert"]["model"](
            encoded_ids, output_hidden_states=True
        ).hidden_states
        return output[-1][0]

    def init_w2v(self, docs: list, vector_size: int = 100) -> None:
        """Initializes word2vec model with docs

        Args:
            docs (list(str)): List of documents as strings
            vector_size (int, optional): Dimensionality of vector embedding. Defaults to 100.
        """
        self.v2v_size = vector_size
        sentences = []
        for doc in docs:
            sentences.append(simple_preprocess(doc))

        self.w2v = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=4,
        ).wv

    def get_w2v(self, raw: str) -> torch.Tensor:
        """Generate word2vec embedding of any string

        Args:
            raw (str):

        Raises:
            Exception: If init_w2v() is not called with a corpus.
            KeyError: If includes word that was not present

        Returns:
            torch.Tensor: Tensor of size (num words, w2v_size)
        """
        if self.w2v is None:
            raise Exception("w2v is not initialized")
        encoded = []
        for word in simple_preprocess(raw):
            encoded.append(self.w2v[word])

        return torch.Tensor(encoded)

    def purge_w2v(self):
        self.w2v = None

    def get_glove_wiki100(self, raw: str) -> torch.Tensor:
        """Generates glove embedding of dimensionality=100 trained on wikipedia on any string

        Args:
            raw (str):

        Returns:
            torch.Tensor: Tensor of size (num words, 100)
        """
        if self.gw is None:
            self.gw = gensim.downloader.load("glove-wiki-gigaword-100")

        encoded = []
        for word in simple_preprocess(raw):
            encoded.append(self.gw[word])

        return torch.Tensor(encoded)

    def purge_glove_wiki100(self):
        self.gw = None

    # def get_sentence_embeddings(
    #     self,
    #     docs: list,
    #     transformer: str = "sbert_mean",
    #     pretrained_path=None,
    #     pooling: str = "mean",
    # ) -> Any:
    #     # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Generates sequence of sentence embeddings embedding for each string provided in text

    #     Args:
    #         text (str | list): Either str or list(str)
    #         transformer (str, optional): Transformer to perform embedding. Defaults to 'sbert_mean'.

    #     Returns:
    #         torch.Tensor: size either (len(text),num sentences, embedding dimension)
    #         or (len(text),num sentences, max(num words per sentence), embedding dimension)
    #     """
    #     if pretrained_path is not None:
    #         pretrained_model = torch.load(pretrained_path)
    #     else:
    #         pretrained_model = None
    #     embeds = []
    #     doc_lengths = []

    #     for _, doc in enumerate(docs):
    #         embed = self._get_doc_sentence_embeddings(
    #             doc, transformer, pooling, pretrained_model=pretrained_model
    #         )

    #         doc_lengths.append(embed.size()[0])
    #         embeds.append(embed)

    #     padded: torch.Tensor = pad_sequence(embeds, batch_first=True, padding_value=0.0)

    #     return padded, torch.tensor(doc_lengths)

    # def _get_doc_sentence_embeddings(
    #     self,
    #     doc: str,
    #     transformer: str = "sbert_mean",
    #     pooling: str = "mean",
    #     pretrained_model=None,
    # ) -> torch.Tensor:
    #     """Generates sequence of sentence embeddings embedding for any raw string

    #     Args:
    #         raw (str): raw text
    #         transformer (str): transformer to use out of (bert, sbert_mean, sbert_cls, xlnet)
    #     Returns:
    #         torch.Tensor: Tensor of size (num sentences, embedding dimension)
    #     """
    #     if transformer not in self.transformers.keys():
    #         raise ValueError(
    #             f"'transformer' argument must be one of {list(self.transformers.keys())}"
    #         )
    #     self._init(transformer)

    #     sentences = sent_tokenize(doc)

    #     tokenizer = self.transformers[transformer]["tokenizer"]
    #     model = self.transformers[transformer]["model"]

    #     if pretrained_model is not None:
    #         model = pretrained_model

    #     encoded = tokenizer(sentences, return_tensors="pt", padding="longest")

    #     with torch.no_grad():
    #         output = model(**encoded).last_hidden_state

    #     if pooling == "mean":
    #         # Mean Pooling
    #         input_mask_expanded = (
    #             encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
    #         )
    #         sum_embeddings = torch.sum(output * input_mask_expanded, 1)
    #         sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    #         return sum_embeddings / sum_mask
    #     else:
    #         return output[:, 0, :].squeeze()

    # def purge(self, transformer="all"):
    #     if transformer not in self.transformers.keys():
    #         raise ValueError(
    #             f"'transformer' argument must be one of {list(self.transformers.keys()) + ['all']}"
    #         )
    #     if transformer == "all":
    #         for k in self.transformers.keys():
    #             self.transformers[k]["tokenizer"] = None
    #             self.transformers[k]["model"] = None
    #     else:
    #         self.transformers[transformer]["tokenizer"] = None
    #         self.transformers[transformer]["model"] = None


# doc = 'hello there! What\'s up? The market.'

# embedder = Embedder()
# embedded = embedder.get_sentence_embeddings(doc)
# print(embedded.size())

# bert = AutoModel.from_pretrained("bert-base-uncased")

# for n, p in bert.named_parameters():
#     print(n, p.size())
