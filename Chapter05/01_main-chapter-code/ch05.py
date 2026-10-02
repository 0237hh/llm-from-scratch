############################### 5.1 텍스트 생성 모델 평가하기
############################### 5.1.1 GPT를 사용해 텍스트 생성하기
import torch
from previous_chapters import GPTModel

GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    # 문맥 길이를 1024에서 256으로 축소
    "context_length" : 256,
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    # 드롭아웃을 0으로 지정할 수 있으며 많이 사용됨.
    "drop_rate" : 0.1,
    "qkv_bias" : False
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

# GPTModel의 인스턴스 객체를 4장의 generate_text_simple 함수에 적용, 
# 2개의 유틸리티 함수 text_to_token_ids 와 toekn_ids_to_token 함수 생성. 
# 텍스트와 토큰 표현 사이의 변환을 수행
import tiktoken
from previous_chapters import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # .unsqueeze(0) : 배치 차원을 추가
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    # 배치 차원 제거
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("출력 텍스트:\n", token_ids_to_text(token_ids, tokenizer))

############################### 5.1.2 텍스트 생성 손실 계산하기
# 입력 텍스트를 받아 텍스트를 생성하는 전반적인 흐름알아보기
# 2개의 입력 샘플을 미리 토큰 ID로 매핑
inputs = torch.tensor([[16833, 3626, 6100], # ["every effort_moves".
                       [40,1107,588]])           # "I really like]"
# 입력에 대응하는 targets는 모델이 생성할 토큰ID를 담고있음
targets = torch.tensor([[3626,6100,345], # ["effort moves you"]
                        [1107, 588, 11311]]) # " rellay like chocolate"]
# 모델 주입 및 각각 3개의 토큰으로 구성된 입력 샘플 2개에 대한 로짓 벡터 계산
# softmax 함수를 적용해 로짓을 확률 점수(probas)로 변환
# 훈련전이니 그레이디언트 추적 OFF.
with torch.no_grad():
    logits = model(inputs)
# 어휘사전의 각 토큰에 대한 확률
probas = torch.softmax(logits, dim=-1)
print(probas.shape)

# 확률 점수에 argmax 함수를 적용해 토큰ID를 추출하는 방식
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("토큰 ID:\n", token_ids)

# 마지막 단계에서 토큰 ID를 텍스트로 변환
print(f"첫번째 샘플의 타깃: {token_ids_to_text(targets[0], tokenizer)}")
print(f"첫번째 샘플의 출력:"
      f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# 2개의 입력 텍스트에 대해 타깃 토큰에 해당하는 초기 소프트맥스 확률 점수를 출력
text_idx = 0
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
print("텍스트 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
print("텍스트 2:", target_probas_2)

# 두 샘플의 확률 점수 target_probas_1, target_probas_2에 대한 손실 계산
# 확률 점수에 로그 적용
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

# 로그 확률을 평균해 하나의 점수로 생성.
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

# 크로스 엔트로피 함수 적용 전 로짓과 타깃 텐서의 크기 확인
print("로짓 크기:", logits.shape)
print("타깃 크기:",targets.shape)

# 파이토치의 cross_entropy 손실 함수를 위해 처음 두 차원을 결합해 텐서 펼치기
logits_flat = logits.flatten(0,1)
targets_flat = targets.flatten()
print("펼친 로짓:", logits_flat.shape)
print("펼친 타깃:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)

############################### 5.1.3 훈련세트와 검증세트의 손실 계산하기
# 소설 로드 (데이터 셋 로드)
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
# 데이터 셋 로드 후 해당 데이터셋에 있는 문자와 토큰 수 확인
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("문자수:",total_characters)
print("토큰수:", total_tokens)

# 데이터셋을 훈련세트와 검증세트로 분할, 데이터 로더를 사용해 LLM 훈련을 위한 배치준비
# 데이터 분할과 로딩 구현ㅇ르 위해 train_ratio를 0.9로 지정,
# 모델 훈련에 데이터의 90%를 사용하고 남은 10%를 평가를 위한 검증 세트로 사용
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# 2장에서 만든 create_dataloader_v1를 사용해 train_data와 val_data의 데이터 로더 생성
from previous_chapters import create_dataloader_v1
torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
# 데이터 로더 순회하면서 올바르게 만들어졌는지 확인
print("훈련 데이터 로더:")
for x,y in train_loader:
    print(x.shape, y.shape)
print("\n검증 데이터 로더:")
for x,y in val_loader:
    print(x.shape, y.shape)

# 훈련 데이터 로더와 검증 데이터 로더가 반환한 배치로 크로스 엔트로피 손실을 계산하는 유틸리티 함수 구현
def calc_loss_batch(input_batch, target_batch, model, device):
    # 데이터를 지정된 장치(GPU)로 전송
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )
    return loss

# 훈련 손실과 검증 손실을 계산하는 함수
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader)==0:
        return float("nan")
    elif num_batches is None:
        # num_batches가 지정되지 않으면 모든 배치를 순회
        num_batches = len(data_loader)
    else:
        # num_batches가 데이터 로더에 있는 배치 개수보다 크면 
        # 배치 횟수를 데이터 로더에 있는 층 개수로 맞춤
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            # 각 배치의 손실을 더함
            total_loss += loss.item()
        else:
            break
    # 모든 배치의 손실을 평균
    return total_loss / num_batches

# 훈련 데이터 로더와 검증 데이터 로더를 사용해 calc_loss_loader 함수를 직접 실행
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CUDA 지원 GPU가 있다면 코드를 수정하지 않고 GPU에서 LLM 훈련
model.to(device)
# 훈련하는 것이 아니므로 효율성을 위해 그레이디언트 추적 OFF.
with torch.no_grad():
    # device 매개변수를 사용해 LLM 모델과 동일한 장치에 데이터를 로드
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("훈련 손실:", train_loss)
print("검증 손실", val_loss)

############################### 5.2 LLM 훈련하기
# LLM 사전 훈련을 위한 함수
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # 손실과 지금까지 처리한 토큰 수를 추적하기 위해 리스트 초기화
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0,-1

    # 메인 훈련 루프 시작점
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            # 이전 배치 반복에서 얻은 손실 그레이디언트를 초기화
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            # 손실 그레이디언트 계산
            loss.backward()
            # 손실 그레이디언트를 사용해 모델 가중치 업데이트
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # 추가적인 평가 단계
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"에포크 {epoch+1} (Step {global_step:06d}): "
                      f"훈련 손실 {train_loss:.3f}, "
                      f"검증 손실 {val_loss:.3f}")
                
        # 각 에포크 후에 샘플 텍스트 출력
        generate_and_print_sample(
            model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

# evaluate_model 함수
# 모델이 업데이트된 후 모델이 향상되었는지 평가할 수 있도록 훈련 세트 손실과 검증 세트에 대해 손실을 계산
# 모델을 평가 모드로 설정해 그레이디언트 추적과 드롭아웃을 비활성화하고 훈련 세트과 검증 세트에 대해 손실을 계산
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 안정적이고 재현 가능한 결과를 위해 평가하는 동안 드롭아웃을 비활성화
    model.eval()
    # 계산 오버헤드를 줄이기 위해 훈련 과정에서 필요하지 않은 그레이디언트 추적 OFF.
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# generate_text_simple 함수
# 모델이 생성한 구체적인 텍스트 샘플 제공
# 훈련하는 동안 모델의 성능을 평가할 수 있음.
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
                        model=model, idx=encoded, 
                        max_new_tokens=50, context_size=context_size)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # 간단한 출력 포맷
        print(decoded_text.replace("\n", " "))
        model.train()

# AdamW 옵티마이저와 앞서 정의한 train_model_simple 함수를 사용해 GPTModel 객체를 10 에포크 동안 훈련
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    # .parameters() 메서드는 모델에 있는 훈련 가능한 모든 가중치 파라미터를 변환
    model.parameters(),
    lr=0.0004, weight_decay=0.1)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader,optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer)

# 훈련 세트 손실과 검증 세트 손실을 나란히 보여주는 그래프 그리기
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses (epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # y축을 공유하는 두번째 x축 생성
    ax2 = ax1.twiny()
    # 눈금을 정리하기 위해 투명한 그래프 생성
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show

epoch_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epoch_tensor, tokens_seen, train_losses, val_losses)

############################### 5.3 무작위성을 제어하기 위한 디코딩 전략
# 비교적 작은 모델로 추론시 GPU가 필요하지 않음. 모델을 GPU->CPU로 변경
# 훈련 후 드롭아웃과 같은 랜덤 구성 요소를 끄기위해 모델을 평가모델로 전환
model.to("cpu")
model.eval()

# GPT 모델의 객체(model)를 generate_text_simple 함수에 전달해 
# 한번에 하나의 토큰씩 텍스트를 생성
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("출력 텍스트:\n", token_ids_to_text(token_ids, tokenizer))
############################### 5.3.1 온도스케일링
# 확률적 샘플링의 예
vocab = {
    "closer":0,
    "every":1,
    "effort":2,
    "forward":3,
    "inches":4,
    "moves":5,
    "pizza":6,
    "toward":7,
    "you":8
}
inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

# generate_text_sample 함수 안에서 로짓을 softmax 함수를 통해 확률로 변환
# 그 다음 argmax 함수를 통해 생성 토큰에 해당하는 토큰ID를 얻음.
# 이를 역어휘사전을 통해 다시 텍스트로 매핑
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])

# 가장 큰 로짓 값과 이에 해당하는 가장 큰 소프트맥스 확률 점수가 네번째 위치이기 때문에
# 생성될 단어는 forward.
# 확률적 샘플링 과정을 굴션하기 위해 argmax를 파이토치의 multinomial 함수로 변경
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])

# 샘플링을 1000번 수행하는 함수 생성
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
              for i in range(1_000)]
    smapled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(smapled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)

# 온도 스케일링
def softmax_with_temperture(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# 그래프로 확인해보기
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# 원본, 낮은 온도, 높은 온도
tempertures = [1,0.1,5]
scaled_probas = [softmax_with_temperture(next_token_logits, T)
                 for T in tempertures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5,3))
for i, T in enumerate(tempertures):
    rects = ax.bar(x+i * bar_width, scaled_probas[i],
                   bar_width, label=f'Temperature={T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()

############################### 5.3.2 탑-k 샘플링
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("탑-k 로짓", top_logits)
print("탑-k 위치", top_pos)

# 파이토치의 where함수를 사용해 탑-k 토큰 중에서 가장 마지막 토큰보다 작은 로짓값을
# 가진 모든 토큰의 로짓 값을 음의 무한대(-inf)로 지정
new_logits = torch.where(
    # 탑-k 토큰 중 마지막 토큰보다 로짓이 작은 토큰을 찾음
    condition=next_token_logits < top_logits[-1],
    # 찾은 토큰의 로짓을 -inf로 설정
    input=torch.tensor(float('-inf')),
    # 다른 모든 토큰은 로짓값을 그대로 유지
    other=next_token_logits
)
print(new_logits)

# 마지막으로 softmax 함수를 적용해 이를 다음 토큰 확률로 변경
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)

############################### 5.3.3 텍스트 함수 수정하기
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    # for 루프는 이전과 동일. 로짓을 받아 마지막 타임 스텝만 사용
    for _ in range(max_new_tokens):
        idx_cond=idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]

        # top-k 샘플링으로 로짓을 필터링
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        # 온도 스케일링 적용
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        # 온도 스케일링을 사용하지 않는 경우 이전처럼 그리디 디코딩을 사용해 다음 토큰을 선택
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        # EOS 토큰을 만나면 생성을 중단
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# 함수 테스트
torch.manual_seed(123)
token_ids = generate(
    model = model,
    idx = text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k = 25,
    temperature=1.4
)
print("출력 텍스트:\n", token_ids_to_text(token_ids, tokenizer))

############################### 5.4 파이토치로 모델 로드하고 저장하기
# torch.save 함수를 사용해 모델의 층과 파라미터를 매핑한 딕셔너리인 state_dict를 저장
torch.save(model.state_dict(), "model.pth")
# model.pth: state_dict를 저장할 파일이름
# state_dict로 모델 가중치 저장 후 이 가중치를 새로운 모델로 로드 가능
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# torch.save를 사용해 모델과 옵티마이저의 state_dict 내용 모두 저장 가능
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "model_and_optimizer.pth")

# 저장된 데이터를 torch.load로 로드후 load_state_dict 메서드로 모델과 옵티마이저의 상태를 복원
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();

############################### 5-5. 오픈AI에서 사전 훈련된 가중치 로드하기
# gpt_download.py 파일에서 download_and_load_gpt2 함수를 임포트. 이 함수를 사용해 GPT-2 구조설정과 가중치 파라미터를 현제 파이썬 세션에 로드
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

print("설정:", settings)
print("파라미터 딕셔너리키:", params.keys())

# GPT 모델 가중치를 파이썬으로 로드 후 settings와 params 딕셔너리를 GPTModel 객체로 복사
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 기본 설정을 특정 값으로 업데이트
model_name = "gpt2-small (124M)"  # 모델 이름
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
# 객체 초기화
gpt = GPTModel(NEW_CONFIG)
gpt.eval();

# assign 유틸리티 함수를 정의
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"크기가 다릅니다. left: {left.shape}, right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

# 오픈 AI 가중치를 GPTModel의 객체로 로드
import numpy as np

# 위치 임베딩과 토큰 임베딩의 가중치를 params의 값으로 설정
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    # 모델의 트랜스포머 블록을 순회
    for b in range(len(params["blocks"])):
        # np.split 함수를 사용해 어텐션 가중치와 편향 가중치를 쿼리, 키, 값, 세 부분으로 나눔
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # 오픈AI의 원본 GPT-2 모델은 토큰 임베딩의 가중치를 출력 층에 재사용해 
    # 전체 파라미터 개수를 절감하는 가중치 묶기를 사용
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

load_weights_into_gpt(gpt, params)
gpt.to(device);

# generate 함수를 사용해 새로운 텍스트를 생성
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

print("출력 텍스트:\n", token_ids_to_text(token_ids, tokenizer))