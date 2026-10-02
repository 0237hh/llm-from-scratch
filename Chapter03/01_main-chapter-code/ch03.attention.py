####################### 3.3.1 훈련 가능한 가중치가 없는 간단한 셀프 어텐션 메커니즘
# 간소화된 셀프 어텐션 메커니즘을 구현해 가중치와 문맥 벡터를 계산하는 방법 알아보기
import torch
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
       [0.55, 0.87, 0.66],  # journey  (x^2)
       [0.57, 0.85, 0.64], # starts   (x^3)
       [0.22, 0.58, 0.33], # with     (x^4)
       [0.77, 0.25, 0.10], # one      (x^5)
       [0.05, 0.80, 0.55]] # step     (x^6)
)

# 셀프 어텐션을 구현하는 첫 번째 단계
# 어텐션 점수라고 부르는 중간값 w을 계산하는 것.
# 두 번째 입력 토큰을 쿼리 토큰으로 사용.
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

#어텐션 점수 정규화
attn_weights_2_tmp = attn_scores_2/attn_scores_2.sum()
print("어텐션 가중치: ", attn_weights_2_tmp)
print("합: ", attn_weights_2_tmp.sum())

# 어텐션 점수를 정규화하는 소프트맥스 함수의 기본적인 구현
def softmax_native(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weights_2_native = softmax_native(attn_scores_2)
print("어텐션 가중치: ", attn_weights_2_native)
print("합: ", attn_weights_2_native.sum())

# 파이토치가 제공하는 소프트맥스 (활성화)함수
attn_weight_2 = torch.softmax(attn_scores_2, dim=0)
print("어텐션 가중치: ", attn_weight_2)
print("합: ", attn_weight_2.sum())

# 임베딩된 입력 토큰 x(i)와 각 토큰에 해당하는 어텐션 가중치를 모두 곱한 후 모두 더해 문맥벡터 z(2)를 계산.
# 문맥 벡터 z(2)는 입력 벡터의 가중치 합이 되며, 각 입력 벡터와 어텐션 가중치를 곱함.
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weight_2[i]*x_i
print(context_vec_2)

####################### 3.3.2 모든 입력에 대한 어텐션 가중치 계산
# 모든 문맥 벡터를 계산하기 위한 코드
attn_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

# 행렬 곱셈을 사용해도 결과는 동일
attn_scores = inputs @ inputs.T
print(attn_scores)

# 각 행의 값을 모두 더해 1이 되도록 정규화
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# 각 행의 값이 모두 더해서 1이 되는지 확인
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("두 번째 행의 합: ", row_2_sum)
print("모든 행의 합: ", attn_weights.sum(dim=-1))

# 어텐션 가중치와 입력을 행렬 곱셈하여 모든 문맥 벡터를 계산
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)


####################### 3.4.1 단계별로 어텐션 가중치 계산하기
# 두 번째 입력 원소
x_2 = inputs[1]
# 입력 임베딩 크기, d_in = 3
d_in = inputs.shape[1]
# 출력 임베딩 크기, d_out = 2
d_out = 2

# 가중치 행렬 초기화
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 쿼리, 키, 값 벡터를 계산
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
valeu2 = x_2 @ W_value
print(query_2)

# 쿼리에 대한 어텐션 가중치를 계산하기 위해 모든 입력 원소에 대한 키와 값 벡터가 필요
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape: ", keys.shape)
print("values.shape: ", values.shape)

# 어텐션 점수 계산
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# 행렬 곱셈으로 이 계산을 일반화해 모든 어텐션 점수를 계산할 수 있음
# 주어진 쿼리에 대한 모든 어텐션 점수
attn_scores_2 = query_2 @ keys.T 
print(attn_scores_2)

# 어텐션 점수의 가중치 구하기
# 어텐션 점수를 소프트맥스 함수로 정규화하여 어텐션 가중치를 구함
# 어텐션 점수를 키의 임베딩 차원의 제곱근으로 나눔
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

# 문맥 벡터 계산
context_vec_2 = attn_weight_2 @ values
print(context_vec_2)

####################### 3.4.2 셀프 어텐션 파이썬 클래스 구현하기
import torch.nn as nn
class SelfAttention_V1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_V1(d_in, d_out)
print(sa_v1(inputs))

# 파이토치 Linear 층을 사용한 셀프 어텐션 클래스
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores/keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

# 사용
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs)) 

####################### 3.5.1 코잘 어텐션 마스크 적용하기
# 소프트 맥스 함수를 사용해 어텐션 가중치를 계산
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

# 파이토치의 tril 함수로 주대각선 위의 값이 0인 마스크를 생성
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

# 해당 마스크와 어텐션 가중치를 곱해 주대각선 위의 값을 0으로 만듬
masked_simple = attn_weights * mask_simple
print(masked_simple)

# 마지막으로 어텐션 가중치를 합이 1이 되도록 다시 정규화.
# 각 행의 합으로 행의 원소를 나누면 됨
row_nums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_nums
print(masked_simple_norm)

# 주대각선 위의 값이 1인 마스크를 만들고 1을 음의 무한대 값(-inf)으로 바꾸는 식으로 더 효율적으로 마스킹할 수 있음
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

# 마스킹된 결과에 소프트 맥스 함수를 적용
attn_weights = torch.softmax(masked/keys.shape[-1]**0.5, dim=1)
print(attn_weights)

####################### 3.5.2 드롭아웃으로 어텐션 가중치에 추가적으로 마스킹하기
# 어텐션 가중치의 절반을 마스킹하기 위해 드롭아웃 비율을 50%fh wlwjd
torch.manual_seed(123)
# 드롭아웃 비율을 50%로 지정
dropout = torch.nn.Dropout(0.5)
# 1로 채워진 행렬 생성
example = torch.ones(6,6)
# 어텐션 가중치 행렬에 50%의 비율로 드롭아웃을 적용하면 행렬에 있는 원소 절반이 랜덤하게 0으로 바뀜
# 삭제된 값을 남은 원소들로 보상하기 위해 행렬에서 남은 원소의 값을 1/0.5 = 2배로 늘림.
# 위와 같은 보상은 어텐션 가중치의 균형을 유지하는데 중요.
print(dropout(example))

#어텐션 가중치 행렬에 드롭아웃을 적용
torch.manual_seed(123)
print(dropout(attn_weights))

####################### 3.5.2 드롭아웃으로 어텐션 가중치에 추가적으로 마스킹하기
# SelfAttention 클래스에 코잘 어텐션과 드롭아웃 기능 추가
# 배치 입력 시뮬레이션을 위한 입력 텍스트 샘플을 중복해서 사용
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

# 코잘 어텐션 클래스
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 드롭아웃 층 추가
        self.dropout = nn.Dropout(dropout)
        # register_buffer 메서드 호출도 추가
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1))

        # 첫번째 차원인 배치 차원은 그대로 유지하면서 두번째 차원과 세번째 차원을 바꿈
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        # 파이토치에서는 밑줄 문자로 끝나는 메서드는 불필요한 메모리 복사를 
        # 피하기 위해 인플레이스 연산을 수행
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

# 사용예시
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape: ", context_vecs.shape)

####################### 3.6 싱글 헤드 어텐션을 멀티 헤드 어텐션으로 확장하기
# 3.6.1 여러 개의 싱글 헤드 어텐션 층 쌓기
# CausalAttention 모듈을 여러 개 쌓아 MultiHeadAttention 클래스 구현
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, 
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
# MultiHeadAttentionWrapper 클래스 사용
torch.manual_seed(123)
context_length = batch.shape[1] # 토큰 개수
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape: ", context_vecs.shape)

# 3.6.2 가중치 분할로 멀티 헤드 어텐션 구현하기
class MutltiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert(d_out % num_heads==0), \
        "d_out은 num_heads노 나누어 떨어져야 함"

        self.d_out = d_out
        self.num_heads = num_heads
        # 원하는 출력 차원에 맞도록 투영 차원을 낮춤
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Linear 층을 사용해 헤드의 출력을 결합
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # 텐서 크기: (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # num_heads 차원을 추가함으로써 암묵적으로 행렬을 분할.
        # 마지막 차원을 num_heads에 맞춰 채움
        # (b, num_tokens, d_out) -> (b_num, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_tokens, self.num_heads, self.head_dim) 크기를
        # (b, self.num_heads, num_tokens, self.head_dim) 크기로 변경
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        # 각 헤드에 대해 점곱을 수행
        attn_scores = queries @ keys.transpose(2,3)
        # 토큰 개수로 마스크를 자름
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 마스크를 사용해 어텐션 점수를 채움
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 텐서크기(b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1,2)

        # 헤드 결합.
        # self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.dropout) 

        # 선형 투영 추가
        context_vec = self.out_proj(context_vec)
        return context_vec