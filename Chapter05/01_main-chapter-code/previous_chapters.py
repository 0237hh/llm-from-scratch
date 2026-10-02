# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# "Build a Large Language Model From Scratch"의 소스 코드
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 코드: https://github.com/rasbt/LLMs-from-scratch
#
# 이 파일은 지금까지 2~4장에서 다룬 모든 관련 코드를 모아놓은 파일입니다.
# 독립적인 스크립트로 실행할 수 있습니다.

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 전체 텍스트를 토큰화합니다.
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 슬라이딩 윈도우를 사용하여 책의 텍스트를
        # max_length 크기의 서로 겹치는 시퀀스로 나눕니다.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 토크나이저를 초기화합니다.
    tokenizer = tiktoken.get_encoding("gpt2")

    # 데이터셋을 생성합니다.
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # 원하는 출력 차원에 맞추기 위해 투영 차원을 각 헤드의 차원으로 나눕니다.
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 각 어텐션 헤드의 출력을 결합하기 위한 선형 계층
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        # 미래 토큰을 참조하지 못하도록 하는 인과적 마스크(causal mask)를 생성합니다.
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # 형태: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # num_heads 차원을 추가하여 행렬을 암묵적으로 여러 헤드로 분할합니다.
        # 마지막 차원을 펼칩니다:
        # (b, num_tokens, d_out)
        # -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 전치:
        # (b, num_tokens, num_heads, head_dim)
        # -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 스케일드 닷 프로덕트 어텐션(scaled dot-product attention)을 계산합니다.
        # 여기에 인과적 마스크(causal mask)를 적용합니다.
        attn_scores = queries @ keys.transpose(2, 3)  # 각 헤드마다 내적을 계산합니다.

        # 원래의 마스크를 현재 토큰 수에 맞게 잘라내고
        # boolean 타입으로 변환합니다.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 마스크가 적용되는 어텐션 점수를 -무한대로 설정합니다.
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 어텐션 점수를 스케일링한 후 softmax를 적용하여 어텐션 가중치를 계산합니다.
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
        )

        attn_weights = self.dropout(attn_weights)

        # 형태:
        # (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 여러 헤드의 결과를 하나로 결합합니다.
        # self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)

        # 선택적인 선형 투영을 적용합니다.
        context_vec = self.out_proj(context_vec)

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 어텐션 블록을 위한 숏컷 연결(shortcut connection)
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # 형태: [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # 원래 입력을 다시 더합니다.

        # 피드 포워드 블록을 위한 숏컷 연결(shortcut connection)
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 원래 입력을 다시 더합니다.

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # 형태: [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx는 현재 문맥에 포함된 인덱스 배열이며 형태는 (B, T)입니다.
    for _ in range(max_new_tokens):

        # 현재 문맥이 지원되는 context_size보다 길어지면 현재 문맥을 잘라냅니다.
        # 예: LLM이 5개의 토큰만 지원하고 문맥 크기가 10이라면
        # 마지막 5개의 토큰만 문맥으로 사용합니다.
        idx_cond = idx[:, -context_size:]

        # 예측값을 가져옵니다.
        with torch.no_grad():
            logits = model(idx_cond)

        # 마지막 시점의 결과만 사용합니다.
        # (batch, n_token, vocab_size)
        # -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # 가장 높은 logits 값을 가진 어휘 항목의 인덱스를 가져옵니다.
        idx_next = torch.argmax(
            logits, dim=-1, keepdim=True
        )  # 형태: (batch, 1)

        # 예측한 인덱스를 현재 시퀀스에 추가합니다.
        idx = torch.cat(
            (idx, idx_next), dim=1
        )  # 형태: (batch, n_tokens+1)

    return idx


if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 어휘 크기
        "context_length": 1024,  # 문맥 길이
        "emb_dim": 768,          # 임베딩 차원
        "n_heads": 12,           # 어텐션 헤드 수
        "n_layers": 12,          # 트랜스포머 레이어 수
        "drop_rate": 0.1,        # 드롭아웃 비율
        "qkv_bias": False        # Query-Key-Value 바이어스 사용 여부
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # 드롭아웃을 비활성화합니다.

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)