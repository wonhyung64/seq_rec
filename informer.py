#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# 1. ProbSparse Self-Attention (O(L²) → O(L logL)로 연산량 감소)
###############################################################################
class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(ProbSparseSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, _ = x.shape

        # Query, Key, Value 생성
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        # Dot-Product Similarity
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5

        # ProbSparse: 가장 중요한 Query-Key 쌍만 남기기 (상위 k개 선택)
        top_k = max(1, int(length * 0.1))  # 10% 만 유지
        sparse_scores, _ = torch.topk(scores, k=top_k, dim=-1)

        # Softmax 및 Dropout
        attn_weights = F.softmax(sparse_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Self-Attention 적용
        output = torch.matmul(attn_weights, v)

        # 차원 변환 및 출력
        output = output.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        output = self.out_proj(output)

        return output


###############################################################################
# 2. Self-Attention Distilling (메모리 사용량 절반으로 감소)
###############################################################################
class SelfAttentionDistilling(nn.Module):
    def __init__(self):
        super(SelfAttentionDistilling, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, length, d_model) → (batch, d_model, length)
        x = self.pool(x)  # 길이를 절반으로 줄임
        x = x.transpose(1, 2)  # 다시 원래 차원으로 변환
        return x


###############################################################################
# 3. Encoder Layer
###############################################################################
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = ProbSparseSelfAttention(d_model, num_heads, dropout)
        self.distilling = SelfAttentionDistilling()
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.attn(x)
        x = self.norm1(x + residual)

        # Self-Attention Distilling 적용 (길이를 절반으로 줄임)
        x = self.distilling(x)

        residual = x
        x = self.feedforward(x)
        x = self.norm2(x + residual)

        return x


###############################################################################
# 4. Generative Decoder Layer
###############################################################################
class GenerativeDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(GenerativeDecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output):
        residual = x
        x, _ = self.self_attention(x, x, x)
        x = self.norm1(x + residual)

        residual = x
        x, _ = self.cross_attention(x, enc_output, enc_output)
        x = self.norm2(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.norm3(x + residual)

        return x


###############################################################################
# 5. Informer Model (Encoder + Generative Decoder)
###############################################################################
class Informer(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, num_enc_layers=2, num_dec_layers=1, dropout=0.1):
        super(Informer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

        # Encoder: 여러 개의 Encoder Layer 쌓기
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, dropout=dropout) for _ in range(num_enc_layers)]
        )

        # Decoder: 여러 개의 Generative Decoder Layer 쌓기
        self.decoder_layers = nn.ModuleList(
            [GenerativeDecoderLayer(d_model, dropout=dropout) for _ in range(num_dec_layers)]
        )

    def forward(self, x_enc, x_dec):
        x_enc = self.input_proj(x_enc)  # 입력 차원 변환
        x_dec = self.input_proj(x_dec)

        # Encoder 처리
        for layer in self.encoder_layers:
            x_enc = layer(x_enc)

        # Generative 방식의 Decoder 처리
        for layer in self.decoder_layers:
            x_dec = layer(x_dec, x_enc)

        # 최종 예측 출력
        output = self.output_proj(x_dec)
        return output


###############################################################################
# 6. 모델 실행 예제
###############################################################################
if __name__ == "__main__":
    batch_size = 8
    input_length = 96
    decoder_length = 96
    input_dim = 1
    d_model = 64
    output_dim = 1

    model = Informer(input_dim, d_model, output_dim, num_enc_layers=2, num_dec_layers=1)
    x_enc = torch.randn(batch_size, input_length, input_dim)


    x_dec = torch.randn(batch_size, decoder_length, input_dim)

    x_enc

    q_proj = nn.Linear(d_model, d_model)
    k_proj = nn.Linear(d_model, d_model)
    v_proj = nn.Linear(d_model, d_model)
    out_proj = nn.Linear(d_model, d_model)
    dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, _ = x

        # Query, Key, Value 생성
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        # Dot-Product Similarity
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5

        # ProbSparse: 가장 중요한 Query-Key 쌍만 남기기 (상위 k개 선택)
        top_k = max(1, int(length * 0.1))  # 10% 만 유지
        sparse_scores, _ = torch.topk(scores, k=top_k, dim=-1)

        # Softmax 및 Dropout
        attn_weights = F.softmax(sparse_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Self-Attention 적용
        output = torch.matmul(attn_weights, v)



    output = model(x_enc, x_dec)
    print("Output shape:", output.shape)  # 예상 결과: (batch, decoder_length, output_dim)