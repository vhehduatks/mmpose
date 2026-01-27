# EgoPose 3D 개선 아이디어

> 최종 업데이트: 2026-01-21

---

## 목차

1. [EfficientHeatmapDecoder 개선](#efficientheatmapdecoder-개선)
2. [Dual Backbone Mutual Learning](#dual-backbone-mutual-learning)
3. [구조적 문제: Heatmap의 3D 정보 한계](#구조적-문제-heatmap의-3d-정보-인코딩-한계)
4. [Backbone Feature Fusion](#backbone-feature-fusion)
5. [Attention 기반 Lifting Network](#attention-기반-lifting-network)
6. [Cross Attention 기반 HMD Fusion](#cross-attention-기반-hmd-fusion)

---

## EfficientHeatmapDecoder 개선

### 현재 구조 분석

```
Z [64] → FC → [256] → reshape → [256, 1, 1]
              ↓
      ConvTranspose2d (5 stages)
              ↓
         [64, 32, 32]
              ↓
    Bilinear Upsample + Conv
              ↓
      Heatmap [16, 47, 47]
```

**현재 파라미터**: 1.35M (Original 40M 대비 96.6% 감소)

**문제점**:
1. Z[64]가 FC로만 처리됨 → spatial 구조에 대한 guidance 부족
2. 단순 순차 upsampling → 정보 흐름이 한 방향
3. 각 관절이 동일한 경로로 생성 → 관절별 특성 반영 어려움

---

### 개선 방안 1: AdaIN (Adaptive Instance Normalization) ⭐ 추천

Z를 style로 사용하여 각 layer에 영향을 주는 방식 (StyleGAN 스타일):

```python
class AdaINHeatmapDecoder(nn.Module):
    """StyleGAN 방식: Z가 각 layer의 normalization에 영향"""
    def __init__(self, num_classes=16, input_size=64):
        super().__init__()

        # Learned constant (시작점) - 4x4 spatial
        self.const = nn.Parameter(torch.randn(1, 256, 4, 4))

        # Z → style parameters (각 layer별 scale, shift)
        self.style_fc = nn.ModuleList([
            nn.Linear(input_size, 256 * 2),  # layer 1: scale + shift
            nn.Linear(input_size, 128 * 2),  # layer 2
            nn.Linear(input_size, 64 * 2),   # layer 3
        ])

        # Upsampling layers
        self.conv1 = nn.Conv2d(256, 128, 3, padding=1)  # 4x4 → 8x8
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)   # 8x8 → 16x16
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)    # 16x16 → 32x32
        self.final = nn.Conv2d(32, num_classes, 1)      # → 47x47

    def adain(self, feat, style):
        """Adaptive Instance Normalization"""
        B, C, H, W = feat.shape
        feat = F.instance_norm(feat)
        scale, shift = style.view(B, 2, C).chunk(2, dim=1)
        return feat * scale.squeeze(1).view(B, C, 1, 1) + shift.squeeze(1).view(B, C, 1, 1)

    def forward(self, z):
        B = z.size(0)
        x = self.const.expand(B, -1, -1, -1)  # [B, 256, 4, 4]

        # Stage 1: 4x4 → 8x8
        x = self.adain(x, self.style_fc[0](z))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.conv1(x), 0.2)

        # Stage 2: 8x8 → 16x16
        x = self.adain(x, self.style_fc[1](z))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.conv2(x), 0.2)

        # Stage 3: 16x16 → 32x32
        x = self.adain(x, self.style_fc[2](z))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.conv3(x), 0.2)

        # Final: 32x32 → 47x47
        x = F.interpolate(x, size=(47, 47), mode='bilinear', align_corners=False)
        heatmap = self.final(x)

        return heatmap
```

**장점**:
- Z가 전체 생성 과정에 **지속적으로 영향**
- Learned constant에서 시작 → 더 안정적인 학습
- 파라미터 증가 적음 (~0.1M)

**예상 파라미터**: ~1.5M

---

### 개선 방안 2: PixelShuffle 기반 Upsampling

ConvTranspose2d 대신 PixelShuffle 사용 (checkerboard artifact 감소):

```python
class PixelShuffleHeatmapDecoder(nn.Module):
    """PixelShuffle로 checkerboard artifact 제거"""
    def __init__(self, num_classes=16, input_size=64):
        super().__init__()

        self.fc = nn.Linear(input_size, 256 * 4 * 4)  # 직접 4x4로

        # PixelShuffle: Conv → channel shuffle → spatial increase
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 256 * 4, 3, padding=1),  # 4x channels for 2x upscale
            nn.PixelShuffle(2),  # [256*4, H, W] → [256, 2H, 2W]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )  # 4x4 → 8x8

        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )  # 8x8 → 16x16

        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )  # 16x16 → 32x32

        self.final = nn.Sequential(
            nn.Upsample(size=(47, 47), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 4, 4)
        x = self.up1(x)   # [B, 256, 8, 8]
        x = self.up2(x)   # [B, 128, 16, 16]
        x = self.up3(x)   # [B, 64, 32, 32]
        x = self.final(x) # [B, 16, 47, 47]
        return x
```

**장점**:
- Checkerboard artifact 제거
- 더 부드러운 upsampling
- ConvTranspose2d와 비슷한 파라미터

---

### 개선 방안 3: Joint-wise Parallel Generation

각 관절별로 독립적인 heatmap 생성:

```python
class JointWiseHeatmapDecoder(nn.Module):
    """각 관절이 독립적인 decoder path로 생성"""
    def __init__(self, num_classes=16, input_size=64):
        super().__init__()

        # Z → 관절별 latent 분리
        self.joint_fc = nn.Linear(input_size, num_classes * 32)  # [B, 16*32]

        # 공유 upsampler (파라미터 효율성)
        self.shared_upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 1→2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 2→4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 4→8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 8→16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 16→32
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # 관절별 refinement (각자 다른 가중치)
        self.joint_refine = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(47, 47), mode='bilinear', align_corners=False),
                nn.Conv2d(16, 1, 3, padding=1)
            ) for _ in range(num_classes)
        ])

    def forward(self, z):
        B = z.size(0)
        # Z → 관절별 분리
        joint_z = self.joint_fc(z).view(B, 16, 32)  # [B, 16, 32]

        heatmaps = []
        for i in range(16):
            jz = joint_z[:, i, :].view(B, 32, 1, 1)  # [B, 32, 1, 1]
            feat = self.shared_upsample(jz)          # [B, 16, 32, 32]
            hm = self.joint_refine[i](feat)          # [B, 1, 47, 47]
            heatmaps.append(hm)

        return torch.cat(heatmaps, dim=1)  # [B, 16, 47, 47]
```

**장점**:
- 각 관절의 특성(크기, 분포)을 개별 학습
- 공유 backbone + 관절별 head 구조

**단점**:
- 순차 처리로 속도 저하 가능 (병렬화 필요)

---

### 개선 방안 4: Attention 기반 Upsampling

Self-attention으로 global context 활용:

```python
class AttentionHeatmapDecoder(nn.Module):
    """Low-resolution에서 self-attention으로 global context"""
    def __init__(self, num_classes=16, input_size=64):
        super().__init__()

        self.fc = nn.Linear(input_size, 256 * 4 * 4)  # 직접 4x4로

        # Self-attention at 4x4 (16 tokens - 효율적)
        self.self_attn = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(256)

        # Upsampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4→8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8→16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16→32
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Upsample(size=(47, 47), mode='bilinear', align_corners=False),
            nn.Conv2d(32, num_classes, 3, padding=1),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 4, 4)
        B, C, H, W = x.shape

        # Self-attention on 4x4 = 16 tokens
        x_flat = x.flatten(2).transpose(1, 2)  # [B, 16, 256]
        x_attn, _ = self.self_attn(x_flat, x_flat, x_flat)
        x_attn = self.attn_norm(x_flat + x_attn)  # Residual
        x = x_attn.transpose(1, 2).view(B, C, H, W)

        x = self.upsample(x)
        heatmap = self.final(x)
        return heatmap
```

**장점**:
- 관절 간 관계를 attention으로 모델링
- Low-resolution (4x4)에서 attention → 효율적

---

### 개선 방안 5: UNet-style Skip Connection

Encoder의 intermediate feature를 decoder에 연결:

```
현재 Encoder 구조:
Heatmap [16, 47, 47]
    ↓ conv1 (stride 2)
[64, 24, 24]  ← skip1
    ↓ conv2 (stride 2)
[128, 12, 12] ← skip2
    ↓ conv3 (stride 2)
[256, 6, 6]   ← skip3
    ↓ flatten + FC
Z [64]

개선된 Decoder:
Z [64]
    ↓ FC + reshape
[256, 6, 6]
    ↓ ConvT + concat(skip3)
[128, 12, 12]
    ↓ ConvT + concat(skip2)
[64, 24, 24]
    ↓ ConvT + concat(skip1)
[32, 47, 47]
    ↓ Conv
Heatmap [16, 47, 47]
```

**장점**:
- Encoder 정보 재사용으로 reconstruction 정확도 향상
- Low-level detail 보존

**단점**:
- Encoder 수정 필요 (intermediate features 반환)

---

### 개선 방안 비교

| 방안 | 구현 난이도 | 예상 효과 | 파라미터 증가 | 추천 순위 |
|------|------------|----------|--------------|----------|
| **AdaIN** | 중간 | ⭐⭐⭐ | ~0.1M | **1** |
| **PixelShuffle** | 낮음 | ⭐⭐ | 없음 | **2** |
| Attention (low-res) | 중간 | ⭐⭐⭐ | ~0.2M | 3 |
| Joint-wise | 높음 | ⭐⭐ | ~0.5M | 4 |
| UNet Skip | 높음 | ⭐⭐⭐ | 없음 | 5 |

**추천**: AdaIN + PixelShuffle 조합

---

### 구현 우선순위

1. **AdaIN 기반 decoder** - Z가 전체 생성에 영향
2. **PixelShuffle** - checkerboard artifact 제거
3. 필요시 Attention 추가

---

## Dual Backbone Mutual Learning

서로 다른 Pretrained Weights (COCO, MPII)를 사용하는 Dual Backbone 구조에서의 Knowledge Transfer 전략.

### 현재 구현 분석

```python
# custom_egopose_baselinel1_head_multi_backbone.py
loss_backbone_latant = MSE(backbone_feat, backbone_feat2)
loss_backbone_heatmap = MSE(final_heatmap, final_heatmap2)
```

### 문제점
1. **Diversity 손실**: 두 feature를 동일하게 만들려고 함
2. **Pretrained 강점 상실**: COCO/MPII 각각의 고유한 knowledge 소멸
3. **방향성 부재**: 양방향 gradient로 어느 쪽이 teacher인지 불명확

---

### 1. Ensemble Teacher (추천)

두 backbone의 출력을 앙상블하여 더 나은 pseudo-teacher 생성.

```
┌───────────────┐     ┌───────────────┐
│  COCO Backbone│     │  MPII Backbone│
│    (feat1)    │     │    (feat2)    │
└───────┬───────┘     └───────┬───────┘
        │                     │
        ▼                     ▼
┌─────────────────────────────────────┐
│     Ensemble (Weighted Average)      │
│  feat_ensemble = w1*feat1 + w2*feat2 │
└───────────────────┬─────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   KD to feat1             KD to feat2
```

```python
def ensemble_mutual_learning_loss(feat1, feat2, heatmap1, heatmap2):
    # Confidence 기반 동적 가중치
    conf1 = heatmap1.max(dim=-1)[0].max(dim=-1)[0].mean()
    conf2 = heatmap2.max(dim=-1)[0].max(dim=-1)[0].mean()

    w1 = conf1 / (conf1 + conf2 + 1e-6)
    w2 = conf2 / (conf1 + conf2 + 1e-6)

    # Ensemble feature (detach로 gradient 차단)
    feat_ensemble = w1 * feat1.detach() + w2 * feat2.detach()

    # 각 backbone이 ensemble을 모방
    loss1 = F.mse_loss(feat1, feat_ensemble)
    loss2 = F.mse_loss(feat2, feat_ensemble)

    return loss1 + loss2
```

---

### 2. Progressive Warmup

초기에는 pretrained 보존, 점진적으로 mutual learning 도입.

```python
def get_mutual_loss_weight(epoch, warmup_epochs=5, rampup_epochs=10):
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs + rampup_epochs:
        return (epoch - warmup_epochs) / rampup_epochs
    else:
        return 1.0
```

---

### 3. Heatmap KL Divergence

Deep Mutual Learning 논문 방식:

```python
def heatmap_kl_divergence_loss(heatmap1, heatmap2, temperature=4.0):
    B, K, H, W = heatmap1.shape

    h1_flat = heatmap1.view(B, K, -1) / temperature
    h2_flat = heatmap2.view(B, K, -1) / temperature

    p1 = F.softmax(h1_flat, dim=-1)
    p2 = F.softmax(h2_flat, dim=-1)

    kl_1_2 = F.kl_div(p1.log(), p2, reduction='batchmean')
    kl_2_1 = F.kl_div(p2.log(), p1, reduction='batchmean')

    return (kl_1_2 + kl_2_1) / 2 * (temperature ** 2)
```

---

### Dual Backbone 실험 우선순위

| 순위 | 방법 | 이유 |
|------|------|------|
| 1 | Ensemble Teacher | 구현 간단, 효과 검증됨 |
| 2 | Progressive Warmup | 기존 방식에 쉽게 추가 가능 |
| 3 | Heatmap KL Div | 논문 방식 재현 |

---

## 구조적 문제: Heatmap의 3D 정보 인코딩 한계

### 핵심 문제

```
Backbone feat [2048, 8, 8]  ← 풍부한 3D 정보 (texture, context, depth cues)
       ↓ (Deconv)
Heatmap [16, 47, 47]        ← 2D 위치 정보만 남음 (depth 손실!)
       ↓ (CNN Encoder)
Z [64]                      ← 극단적 압축 (추가 손실)
       ↓
3D Pose                     ← depth 정보 부족으로 ambiguity
```

### 학술적 근거

1. **Depth Ambiguity 문제**: "하나의 2D pose가 여러 3D pose로 매핑될 수 있음"
2. **CNN Encoder의 정보 손실**: "CNN 기반 인코더가 heatmap 정보를 제대로 보존하지 못함"
3. **Lifting by Image**: "이미지의 풍부한 semantic/texture 정보가 더 정확한 lifting에 기여"

### 결론

**Heatmap은 2D 위치 인코딩에 최적화**되어 있으며, 3D depth 정보를 담기 어려움.
→ **Backbone feature를 별도로 활용**하여 depth cues 보존 필요.

---

## Backbone Feature Fusion

### 방안 A: Backbone + Heatmap Latent Concat (추천)

```
Backbone feat [2048, 8, 8]
       │
       ├──→ GAP → [2048] → FC → [256]  ← Backbone latent
       │                        │
       ↓ (Deconv)               │
Heatmap [16, 47, 47]            │
       ↓ (Encoder)              │
Z_hm [64]                       │
       │                        │
       └──── Concat ────────────┘
              ↓
         [64 + 256 + 64(HMD)] = [384]
              ↓
         Pose Decoder → 3D Pose
```

### 방안 B: 2D Coords + Backbone Feature

```
Heatmap → soft-argmax → 2D coords [16, 2] + conf [16]
                              │
Backbone feat → GAP → FC → Context [256]
                              │
              Concat ─────────┘
                 ↓
           [32 + 16 + 256 + 9(HMD)] = 313
                 ↓
           Lifting Network → 3D Pose
```

---

## Attention 기반 Lifting Network

### Cross-Attention (2D → Backbone) ⭐ 추천

각 관절의 2D 위치가 Backbone feature에서 해당 depth 정보를 쿼리:

```
Query: 2D joint tokens [B, 16, D]    ← "이 2D 위치의 depth는?"
Key/Value: Backbone tokens [B, 64, D] ← "spatial depth 정보"
                    ↓
            Cross-Attention
                    ↓
      Depth-aware joint features [B, 16, D]
                    ↓
                3D Pose
```

```python
class CrossAttentionLifting(nn.Module):
    def __init__(self, joint_dim=64, backbone_dim=256, num_heads=4):
        super().__init__()

        self.joint_embed = nn.Linear(2, joint_dim)
        self.backbone_proj = nn.Conv2d(2048, backbone_dim, 1)
        self.kv_proj = nn.Linear(backbone_dim, joint_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim, num_heads=num_heads, batch_first=True)

        self.output_proj = nn.Linear(joint_dim, 3)

    def forward(self, coords_2d, backbone_feat, confidence, hmd_info):
        # Joint queries from 2D coords
        joint_q = self.joint_embed(coords_2d)  # [B, 16, 64]

        # Backbone spatial tokens
        backbone_tokens = self.backbone_proj(backbone_feat).flatten(2).transpose(1, 2)
        backbone_kv = self.kv_proj(backbone_tokens)  # [B, 64, 64]

        # Cross attention
        depth_features, attn_weights = self.cross_attn(
            query=joint_q, key=backbone_kv, value=backbone_kv)

        pose_3d = self.output_proj(depth_features)
        return pose_3d, attn_weights
```

**장점**:
1. 선택적 depth 쿼리: 각 관절이 필요한 spatial 위치에서 depth 정보
2. 해석 가능: `attn_weights`로 어느 위치 참조했는지 시각화
3. 역할 분리: 2D=Query(위치), Backbone=Key/Value(depth)

---

## Cross Attention 기반 HMD Fusion

### Joint-wise Cross Attention (추천)

각 관절이 독립적으로 HMD 정보에 attend:

```python
class JointHMDCrossAttention(nn.Module):
    def __init__(self, joint_dim=64, num_heads=4):
        super().__init__()

        # HMD를 3개 token으로 (head, right_hand, left_hand)
        self.hmd_embed = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * joint_dim)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim, num_heads=num_heads, batch_first=True)

    def forward(self, joint_features, hmd_info):
        B = joint_features.size(0)
        hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)  # [B, 3, D]

        attn_out, _ = self.cross_attn(
            query=joint_features,  # [B, 16, D]
            key=hmd_tokens,        # [B, 3, D]
            value=hmd_tokens)

        return joint_features + attn_out  # Residual
```

**효과**: 손 관절 → 손 HMD, 몸통 → head HMD로 자연스러운 매핑

---

## Hybrid Lifting: Baseline + Attention Refinement ⭐ NEW

> 2026-01-27 추가

### 동기

| 모델 | MPJPE | 특징 |
|------|-------|------|
| **Baseline** | **41.37mm** 🏆 | Conv Encoder + Linear, 안정적 |
| ViT Lifting v3 | 45.34mm | Full Attention, 학습 불안정 |

**목표**: Baseline의 안정성 + ViT의 관절 관계 모델링 결합

---

### 방안 비교

| 방안 | 변경점 | 리스크 |
|------|--------|--------|
| 1. Attention HMD Fusion | HMD 융합만 변경 | 최소 |
| 2. Joint-wise Feature | Per-joint pooling | 중간 |
| **3. Conv + Attention Refinement** | **Conv Encoder 유지 + Attention 보완** | **낮음** |

---

### 방안 3 상세 설계 (선택)

```
Heatmap [16, 47, 47]
         ↓
┌─────────────────────────────────────────┐
│  Conv Encoder (Baseline 동일)           │
│  Conv: 16→64→128→256, GAP → 64-dim      │
└─────────────────────────────────────────┘
         ↓
      Z [B, 64]
         ↓
┌─────────────────────────────────────────┐
│  Z Reshape: [B, 64] → [B, 16, 4]        │
│  (관절당 4-dim latent)                  │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Joint Embedding: [B, 16, 4] → [B, 16, D] │
│  Linear(4 → joint_dim)                  │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Self-Attention (관절 간 관계)          │
│  Query/Key/Value: [B, 16, D]            │
│  Output: [B, 16, D]                     │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  HMD Cross-Attention                    │
│  Query: Joint tokens [B, 16, D]         │
│  Key/Value: HMD tokens [B, 3, D]        │
│  (head, right_hand, left_hand)          │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Output Head                            │
│  Linear(D → 3) per joint                │
│  Output: [B, 16, 3]                     │
└─────────────────────────────────────────┘
```

---

### 핵심 설계 원칙

1. **Conv Encoder 재사용**: Baseline의 검증된 heatmap→latent 변환
2. **Z 분해**: 64-dim을 16관절 × 4-dim으로 재해석
3. **Self-Attention**: 관절 간 관계 (skeleton structure)
4. **Cross-Attention HMD**: 관절별 HMD 정보 선택적 참조
5. **LinearModel 제거**: Attention이 대체

---

### 구현 코드 (CustomEgoposeHybridLiftingHead)

```python
class HybridLiftingModule(nn.Module):
    """Baseline Conv Encoder + Attention Refinement"""

    def __init__(self, num_joints=16, latent_dim=64, joint_dim=64,
                 num_heads=4, num_self_attn_layers=2, dropout=0.1):
        super().__init__()

        # Z를 관절별로 분해: 64 = 16 * 4
        self.latent_per_joint = latent_dim // num_joints  # 4

        # Joint embedding: 4 → joint_dim
        self.joint_embed = nn.Linear(self.latent_per_joint, joint_dim)

        # Positional encoding for 16 joints
        self.pos_embed = nn.Parameter(torch.randn(1, num_joints, joint_dim) * 0.02)

        # Self-Attention layers
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=joint_dim, nhead=num_heads,
                dim_feedforward=joint_dim*4, dropout=dropout, batch_first=True
            ) for _ in range(num_self_attn_layers)
        ])

        # HMD embedding: 9 → 3 tokens
        self.hmd_embed = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * joint_dim)
        )

        # Cross-Attention: joints query HMD
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(joint_dim)

        # Output projection
        self.output_proj = nn.Linear(joint_dim, 3)

    def forward(self, z, hmd_info):
        B = z.size(0)

        # Z → joint tokens: [B, 64] → [B, 16, 4] → [B, 16, D]
        z_joints = z.view(B, -1, self.latent_per_joint)  # [B, 16, 4]
        joint_tokens = self.joint_embed(z_joints)  # [B, 16, D]
        joint_tokens = joint_tokens + self.pos_embed

        # Self-Attention (관절 간 관계)
        for layer in self.self_attn_layers:
            joint_tokens = layer(joint_tokens)

        # HMD tokens: [B, 9] → [B, 3, D]
        hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)

        # Cross-Attention (joints ← HMD)
        cross_out, _ = self.cross_attn(
            query=joint_tokens, key=hmd_tokens, value=hmd_tokens
        )
        joint_tokens = self.cross_norm(joint_tokens + cross_out)

        # Output: [B, 16, 3]
        pose_3d = self.output_proj(joint_tokens)

        return pose_3d
```

---

### Loss Functions (Baseline 동일)

| Loss | Weight | 역할 |
|------|--------|------|
| `loss_kpt` (MSE) | 1000 | 2D heatmap supervision |
| `loss_heatmap_recon` (MSE) | 250 | Heatmap reconstruction |
| `loss_pose_l2norm` | 1.0 | 3D pose L2 distance |
| `loss_cosine_similarity` | 0.1 | 방향 유사도 |
| `loss_limb_length` | 0.25 | 팔다리 길이 일관성 |
| `loss_hmd` (MSE) | 1.0 | HMD reconstruction |

---

### 예상 효과

| 요소 | Baseline | Hybrid | 효과 |
|------|----------|--------|------|
| Heatmap→Z | Conv Encoder | **동일** | 안정성 유지 |
| Z 활용 | Global (64) | Per-joint (16×4) | 관절별 정보 분리 |
| 관절 관계 | Linear (implicit) | Self-Attn (explicit) | Skeleton 구조 학습 |
| HMD 융합 | Add | Cross-Attn | 선택적 참조 |

**예상**: Baseline 수준 안정성 + 관절 관계 모델링 → 41mm 이하 목표

---

## 참고 논문

1. **Deep Mutual Learning** (Zhang et al., 2017) - arXiv:1706.00384
2. **Knowledge Distillation** (Hinton et al., 2015) - arXiv:1503.02531
3. **A Simple Baseline for 3D Pose** (Martinez et al., 2017) - ICCV 2017
4. **StyleGAN** (Karras et al., 2019) - CVPR 2019
5. **Lifting by Image** - arXiv:2312.15636

---

> **실험 결과**: `EXPERIMENT_RESULTS.md` 참조
> - Single COCO Baseline: **41.37mm** 🏆
> - 목표: 41mm 이하 달성
