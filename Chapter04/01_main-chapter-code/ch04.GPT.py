##################### 4.1 LLM 구조 구현하기
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 어휘사전 크기
    "context_length": 1024, # 문맥 길이
    "emb_dim": 768,         # 임베딩 차원
    "n_heads": 12,          # 어텐션 헤드 개수
    "n_layers": 12,         # 층 개수
    "drop_rate": 0.1,       # 드롭아웃 비율
    "qkv_bias": False       # 쿼리, 키, 값 계산을 위한 편향
}

# 1. 더미 GPT백본 모델인 DummyGPTModel 클래스 구현하기
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 더미 트랜스포머 블록 사용
        self.trf_block = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# 나중에 실제 트랜스포머 블로긍로 교체될 간단한 터미 클래스
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    # 아무것도 하지않고 입력을 그냥 반환
    def forward(self, x):
        return x

# 나중에 실제 층 정규화를 위한 층으로 교체될 간단한 더미 클래스
class DummyLayerNorm(nn.Module):
    # 층 정규화 인터페이스를 흉내내기 위한 매개변수
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    def forward(self, x):
        return x

# tiktoken 토크나이저로 GPT 모델에 사용할 2개의 텍스트로 구성된 배치를 토큰화
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

# 1억 2,400백만 파라미터 크기의 DummyGPTModel 모델을 초기화하고 토큰화된 batch를 주입
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("출력크기: ", logits.shape)
print(logits)

##################### 4-2. 층 정규화로 활성화 정규화하기
# 4개의 차원(특성)을 가진 2개의 샘플 생성
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out = layer(batch_example)
print(out)

# 층 정규화 적용전에 평균과 분산 확인
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("평균:\n", mean)
print("분산:\n", var)

# 층 정규화 적용
# 입력 값에서 평균을 빼고 분산의 제곱근(표준 편차)로 나눔
out_norm = (out-mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.mean(dim=-1, keepdim=True)
print("정규화된 층 출력:\n", out_norm)
print("평균:\n", mean)
print("분산:\n", var)

# 층 정규화 클래스 구현
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift

# LayerNorm 모듈을 배치에 적용
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var (dim=-1, unbiased=False, keepdim=True)
print("평균:\n", mean)
print("분산\n", var)

##################### 4.3 LLM 구조 구현하기
# GELU 활성화 함수 구현
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1+torch.tanh (
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.44715 * torch.pow(x,3))
        ))

import matplotlib.pyplot as plt
gelu, relu = GELU(), nn.ReLU()

# -3에서 3 사이에서 100개의 데이터 포인트를 만듬
x = torch.linspace(-3,3,100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8,3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]),1):
    plt.subplot(1,2,i)
    plt.plot(x,y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()

# FeedForward: GELU 함수를 사용해 LLM의 트랜스포머 블록에 사용할 작은 신경망 모듈
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self, x):
        return self.layers(x)

# FeedForward 모듈 초기화
ffn = FeedForward(GPT_CONFIG_124M)
# 배치 차원이 2인 샘플 입력 생성
x = torch.rand(2,3,768)
out = ffn(x)
print(out.shape)

##################### 4.4 숏컷 연결 추가하기
# forward 메서드에서 숏컷 연결을 추가하는 방법
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        # 5개의 층 생성
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])
    def forward(self, x):
        for layer in self.layers:
            # 현재 층의 출력을 계산
            layer_output = layer(x)
            # 숏컷 연결을 적용할 수 있는지 확인
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

# 숏컷 연결이 없는 신경망 초기화해보기
layer_sizes = [3,3,3,3,3,1]
sample_input = torch.tensor([[1.,0.,-1.]])
# 1. 초기 가중치를 재현할 수 있도록 랜덤 시드 지정
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

# 2. 모델의 역전파에서 그레이디언트를 계산하는 함수를 구현
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    # 타깃과 출력의 가까운 정도를 기반으로 손실을 계산
    loss = loss(output, target)

    # 그레이디언트 계싼ㅇ르 위한 역전파
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}의 평균 그레이디언트는 {param.grad.abs().mean().item()} 입니다.")

# print_gradients 함수를 숏컷 연결이 없는 모델에 적용
print_gradients(model_without_shortcut, sample_input)

##################### 4.5 어텐션과 선형 층을 트랜스포머 블록에 연결하기
# GPT의 트랜스포머 블록
from previous_chapters import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 어텐션 블록을 위한 숏컷 연결
        shortcut = x 
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        # 원본 입력 더하기
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        # 원본 입력 더하기
        x = x + shortcut
        return x

# 트랜스포머 블록 초기화 및 샘플 데이터 전달
torch.manual_seed(123)
x = torch.rand(2,4,768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("입력 크기: ", x.shape)
print("출력 크기: ", output.shape)

##################### 4.6 GPT 모델 구조 구현
# GPT 모델 구조 구현
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
        # 장치 설정을 통해 입력 데이터가 어디에 있는지에 따라 모델을 CPU또는 GPU에서 훈련 가능
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# 배치 텍스트 입력 주입
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("입력 배치:\n", batch)
print("\n출력 크기:\n", out.shape)
print(out)

# numel() 메서드를 사용해 모델 파라미터 텐서에 있는 총 파라미터 개수 확인
total_params = sum(p.numel() for p in model.parameters())
print(f"총 파라미터 개수 : {total_params: ,}")

##################### 4.7 텍스트 생성하기
def generate_text_simple(model, idx, 
                # idx: 현재 문맥이 담긴(batch, num_tokens) 크기의 인덱스 배열
                         max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # 현재 문맥이 모델이 지원하는 문맥 크기를 초과하면 잘라냄.
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits=model(idx_cond)
        # 마지막 타임 스텝만 사용하므로 (batch, num_tokens, vocab_size)가
        # (batch, vocab_size)가 됨.
        logits = logits[:,-1,:]
        # probas의 크기는 (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)
        # idx_next의 크기는 (batch,1).
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # 선택한 인덱스를 현재 시퀀스에 추가하므로 idx의 크기는 (batch, num_tokens+1)가 됨
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx

#generate_text_simple 함수 테스트
# 1. 입력 문맥을 토큰 ID로 인코딩
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("인코딩된 ID: ", encoded)
# 배치 차원을 추가
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape: ", encoded_tensor.shape)

# 2. 모델을 .eval() 모드로 변경. 
model.eval()
out = generate_text_simple(
    model = model,
    idx = encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("출력: ", out)
print("출력 길이: ", len(out[0]))

# 3. 토크나이저의 .decode 메서드를 사용해 ID를 텍스트로 변환
decode_text = tokenizer.decode(out.squeeze(0).tolist())
print(decode_text)