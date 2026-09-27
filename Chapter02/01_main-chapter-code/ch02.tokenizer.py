import os
import urllib.request

# 훈련 작업에 사용할 텍스트 데이터 불러오기
if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("총 문자 개수: ", len(raw_text))
# 파일의 처음 99개 문자 출력
print(raw_text[:99])

# 공백을 기준으로 텍스트를 분할
import re
text = "Hello, world. This, is a test"
# result = re.split(r'(\s)', text)
# print(result)

#공백, 쉼표와 마침표를 분할하도록 정규표현식 수정
result = re.split(r'([,.]|\s)', text)
print(result)

# 중복된 공백문자 제거
result = [item for item in result if item.strip()]
print(result)

# 물음표, 따옴표, 이중 대시와 같은 다른 유형의 구두점과 특수 문자 처리
text2 = "Hello, world. Is this-- a test?"
result2 = re.split(r'([,.:;?_!"()\']|--|\s)', text2)
result2 = [item.strip() for item in result2 if item.strip()]
print(result2)

# 기본적인 토크나이저가 준비되었으므로 이디스 워튼의 소설 전체에 적용
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

# 처음 30개의 토큰 확인
print(preprocessed[:30])

################## 2.3 토큰을 토큰 ID로 변환하기
# 1. 토큰화된 이디스 워튼의 단편 소설이 파이썬 변수 preprocessed에 저장되어 있으므로
# 모든 고유 토큰의 리스트를 만들고 알파벳 순으로 정렬하여 어휘사전 크기 확인
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# 어휘사전 생성 후 처음 51개 항목 출력
vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i>=50:
        break

# 파이썬으로 완전한 토크나이저 구현하기
class SimpleTokenizerV1:
    def __init__(self, vocab):
        # vacab: encode, decode 메서드에서 참조할 수 있도록 어휘사전을 클래스의 속성으로 저장.
        self.str_to_int = vocab 
        # 토큰 ID를 원본 텍스트 토큰으로 매핑하는 역어휘사전 생성
        self.int_to_str = {i:s for s, i in vocab.items()}

    # 입력 텍스트를 처리하여 토큰 ID로 변경
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # 토큰 ID를 텍스트로 되돌림
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # 지정된 구두점 문자 앞의 공백을 삭제
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text 

# 이디스 워튼의 단편 소설의 한 구절을 토큰화
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know, Mrs.Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

# 토큰 ID를 다시 텍스트로 변환
print(tokenizer.decode(ids))

# 특수 토큰 추가하기
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items())) # 1132

# 추가 확인을 위해 수정된 어휘사전에서 마지막 5개의 항목을 출력
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item) 
# ('younger', 1127)
# ('your', 1128)
# ('yourself', 1129)
# ('<|endoftext|>', 1130)
# ('<|unk|>', 1131)

# 알지 못하는 단어를 처리하는 간단한 텍스트 토크나이저
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab 
        self.int_to_str = {i:s for s, i in vocab.items()}

    # 입력 텍스트를 처리하여 토큰 ID로 변경
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int
                        # 알지 못하는 단어를 <|unk|> 토큰으로 변경
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # 토큰 ID를 텍스트로 되돌림
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # 구두점 문자 앞의 공백을 삭제
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) 
        return text 

# 서로 관련이 없는 2개의 독립된 문장을 연결한 간단한 샘플에 사용
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

# 어휘사전과 함께 SimpleTokenizerV2로 토큰화
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))

################## 2.5 BPE (바이트 페어 인코딩)
# titoken 버전 확인
from importlib.metadata import version
import tiktoken
print("tiktoken 버전: ", version("tiktoken"))

# tiktoken 라이브러리에서 BPE 토크나이저를 다음과 같이 초기화할 수 있음
tokenizer = tiktoken.get_encoding("gpt2")

# 토크나이저 사용법
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    " of someunknowunPlace. "
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

# 토크나이저 사용법 (decode)
strings = tokenizer.decode(integers)
print(strings)

# 알지 못하는 단어 Akwirw ier에 적용.
integers = tokenizer.encode("Akwirw ier")
print(integers)

#개별 토큰 아이디 출력
print(tokenizer.encode("Ak"))
print(tokenizer.encode("w"))
print(tokenizer.encode("ir"))
print(tokenizer.encode("w"))
print(tokenizer.encode(" "))
print(tokenizer.encode("ier"))

# decode 메서드 호출
for i in integers:
    print(f"{i} -> {tokenizer.decode([i])}")

print(tokenizer.decode([33901, 86, 343, 86, 220, 959]))


################## 2.6 슬라이딩 윈도로 데이터 샘플링하기
# 슬라이딩 윈도를 사용해 훈련 데이터 셋에서 입력-타깃 쌍을 추출하는 데이터 로더 구현
# 1. BPE(바이트 페어 인코딩) 토크나이저로 소설 전체를 토큰화
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# 2. 데이터 셋에 있는 처음 50개 토큰 삭제
enc_sample = enc_text[50:]

# 3. 다음 단어 예측을 위해 입력-타깃 쌍으로 만드는 가장 쉽고 직관적인 방법: 입력 토큰을 담은 x와 입력에서 토큰 하나만큼 이동한 타깃을 담은 y변수를 만드는 것.
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

# 입력한 토큰 하나만큼 이동시킨 타깃을 사용해 다음 단어 예측 작업을 구성할 수 있음. 
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

# 입력 데이터셋을 순회하면서 파이토치 텐서로 입력과 타깃을 반환하는 데이터 로더 구현
# 데이터 셋 클래스를 위한 코드
import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # 전체 텍스트를 토큰화
        token_ids = tokenizer.encode(txt) 

        # 슬라이딩 윈도를 사용해 책을 max_length 길이의 중첩된 시퀀스로 나눔.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    # 데이터 셋에 있는 전체 행 수를 반환
    def __len__(self):
        return len(self.input_ids)
    # 데이터셋에서 하나의 행을 반환
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# 3. GPTDatasetV1을 사용해 파이토치 DataLoader를 통해 입력을 배치로 로드
# 입력-타깃 쌍의 배치를 생성하기 위한 데이터 로더
def create_data_loader_v1(txt, batch_size=4, max_length=256,
                          stride=128, shuffle=True, drop_last=True,
                          num_workers=0):
    # 토크나이저 초기화
    tokenizer=tiktoken.get_encoding("gpt2")
    # 데이터셋 생성
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # drop_last = True: batch_size보다 작ㅇ르 경우 훈련 손실이 갑자기 높아지는 것을
        # 피하기 위해 마지막 배치를 삭제
        drop_last=drop_last,
        # 전처리에 사용할 CPU프로세서 개수
        num_workers=num_workers
    )
    return dataloader

# 테스트
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
dataloader = create_data_loader_v1(raw_text, batch_size=1, 
                                   max_length=4, stride=1, shuffle=False)
# 데이터 로더를 파이썬 반복자(iterator)로 변형한 다음, 파이썬 내장 next()함수로 다음 원소를 추출
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

# 배치 크기가 1보다 클 경우 데이터 로더로 샘플링하는 방법.
# 배치 사이에 중첩이 있으면 과대적합이 증가할 수 있는데 이렇게 하면 과대적합을 피할 수 있음
dataloader = create_data_loader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("입력: \n", inputs)
print("\n타깃: \n", targets)

################## 2.7 토큰 임베딩 만들기
# 1. 토큰 ID를 임베딩 벡터로 변환하는 방법
inputs_ids = torch.tensor([2,3,5,1])
vocab_size = 6
output_dim = 3
# vocab_size와 output_dim을 사용해 파이토치의 임베딩 층을 초기화할 수 있음.
# 결과를 재현 가능하도록 만들기 위해 랜덤시드를 123으로 지정
torch.manual_seed(13)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# 임베딩 층에 있는 가중치 행렬을 출력
# 임베딩 층의 가중치 행렬의 값은 LLM 최적화의 일부로 LLM훈련 과정에서 최적화 됨.
print(embedding_layer.weight)

# 토큰 ID에 적용해 임베딩 벡터 얻기
print(embedding_layer(torch.tensor([3])))

# 4개의 입력 ID에 모두 적용
print(embedding_layer(inputs_ids))

################## 2.8 단어 위치 인코딩하기
# 입력 토큰을 256차원의 벡터 표현으로 인코딩하기
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# token_embedding_layer를 사용해 데이터 로더를 통해 샘플링한 각 배치에 있는 토큰 ID를 256차원 벡터로 임베딩.
# 배치 크기가 8, 4개의 토큰씩 들어있다면 벡터의 사이즈는 8X4X256이 되어야 함
max_length = 4
dataloader = create_data_loader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("토큰 ID: \n", inputs)
print("\n입력크기:\n", inputs.shape)

# 임베딩 층을 사용해 위 토큰 ID를 256차원 벡터로 임베딩
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# GPT 모델의 절대 임베딩 방법: token_embedding_layer와 동일한 임베딩 차원을 가지는 또 다른 임베딩 층을 생성
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

# 위치 임베딩 벡터는 4개의 256차원의 벡터로 구성됨. 벡터를 토큰 임베딩에 바로 더할 수 있음.
# 파이토치는 4X256 차원의 pos_embeddings 텐서를 배치에 있는 4X256차원의 토큰 임베딩 텐서 8개에 각각 더함
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)