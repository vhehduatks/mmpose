# Dual Backbone 개선 아이디어

서로 다른 Pretrained Weights (COCO, MPII)를 사용하는 Dual Backbone 구조에서의 Knowledge Transfer 전략.

## 현재 구현 분석

### 현재 방식
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

## 개선 아이디어

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

**구현:**
```python
def ensemble_mutual_learning_loss(feat1, feat2, heatmap1, heatmap2):
    """
    Args:
        feat1: COCO backbone features [B, C, H, W]
        feat2: MPII backbone features [B, C, H, W]
        heatmap1, heatmap2: 각 backbone의 heatmap 출력
    """
    # Confidence 기반 동적 가중치
    conf1 = heatmap1.max(dim=-1)[0].max(dim=-1)[0].mean()
    conf2 = heatmap2.max(dim=-1)[0].max(dim=-1)[0].mean()

    w1 = conf1 / (conf1 + conf2 + 1e-6)
    w2 = conf2 / (conf1 + conf2 + 1e-6)

    # Ensemble feature (detach로 gradient 차단 → pseudo-teacher)
    feat_ensemble = w1 * feat1.detach() + w2 * feat2.detach()

    # 각 backbone이 ensemble을 모방
    loss1 = F.mse_loss(feat1, feat_ensemble)
    loss2 = F.mse_loss(feat2, feat_ensemble)

    return loss1 + loss2
```

**장점:**
- 두 pretrained의 강점을 결합
- Confidence 높은 backbone에 더 많은 가중치
- 각 backbone이 "집단 지성"을 학습

---

### 2. Correlation Alignment (Diversity 보존)

Feature **값**이 아닌 **관계(correlation)**만 유사하게.

```python
def correlation_alignment_loss(feat1, feat2):
    """
    두 feature의 correlation structure만 유사하게
    값 자체는 다를 수 있음 → pretrained diversity 보존
    """
    # Flatten: [B, C, H, W] → [B, C, H*W]
    f1 = feat1.view(feat1.size(0), feat1.size(1), -1)
    f2 = feat2.view(feat2.size(0), feat2.size(1), -1)

    # Channel-wise correlation matrix: [B, C, C]
    corr1 = torch.bmm(f1, f1.transpose(1, 2)) / (f1.size(-1) + 1e-6)
    corr2 = torch.bmm(f2, f2.transpose(1, 2)) / (f2.size(-1) + 1e-6)

    return F.mse_loss(corr1, corr2)
```

**장점:**
- 각 backbone의 고유한 feature 값 유지
- "어떻게 attention하는가"만 공유
- Pretrained knowledge 보존

---

### 3. Progressive Warmup

초기에는 pretrained 보존, 점진적으로 mutual learning 도입.

```python
def get_mutual_loss_weight(epoch, warmup_epochs=5, rampup_epochs=10):
    """
    epoch 0-5:   weight = 0 (각자 독립 학습)
    epoch 5-15:  weight = 0→1 (점진적 증가)
    epoch 15+:   weight = 1 (full mutual learning)
    """
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs + rampup_epochs:
        return (epoch - warmup_epochs) / rampup_epochs
    else:
        return 1.0

# 사용
alpha = get_mutual_loss_weight(current_epoch)
loss_mutual = alpha * ensemble_mutual_learning_loss(feat1, feat2, ...)
```

**장점:**
- 초기 학습에서 pretrained knowledge 보존
- 갑작스러운 knowledge 충돌 방지
- 안정적인 학습 곡선

---

### 4. One-way KD (Sub → Main)

Inference에서 main backbone만 사용할 경우, sub가 main을 돕는 구조.

```python
def one_way_kd_loss(feat_main, feat_sub):
    """
    Sub backbone → Main backbone 방향으로만 knowledge transfer
    Sub backbone은 GT에서만 학습
    """
    # Main은 sub의 knowledge를 받음 (sub는 detach)
    loss_kd = F.mse_loss(feat_main, feat_sub.detach())
    return loss_kd
```

**장점:**
- Main backbone 성능 최대화에 집중
- Sub backbone은 auxiliary teacher 역할
- Inference 시 sub 제거 가능

---

### 5. Heatmap KL Divergence (논문 방식 적용)

Deep Mutual Learning 논문의 KL divergence를 heatmap에 적용.

```python
def heatmap_kl_divergence_loss(heatmap1, heatmap2, temperature=4.0):
    """
    Heatmap을 확률 분포로 변환 후 KL divergence
    Temperature scaling으로 soft target 생성
    """
    B, K, H, W = heatmap1.shape

    # Spatial softmax with temperature
    h1_flat = heatmap1.view(B, K, -1) / temperature
    h2_flat = heatmap2.view(B, K, -1) / temperature

    p1 = F.softmax(h1_flat, dim=-1)
    p2 = F.softmax(h2_flat, dim=-1)

    # Symmetric KL divergence
    kl_1_2 = F.kl_div(p1.log(), p2, reduction='batchmean')
    kl_2_1 = F.kl_div(p2.log(), p1, reduction='batchmean')

    return (kl_1_2 + kl_2_1) / 2 * (temperature ** 2)
```

**장점:**
- 논문의 검증된 방식
- Hard target보다 더 많은 정보 전달
- 관절 위치의 "불확실성"도 학습

---

### 6. Attention-based Fusion

학습 가능한 attention으로 두 backbone의 기여도 결정.

```python
class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, feat1, feat2):
        # Concatenate and compute attention weights
        combined = torch.cat([
            F.adaptive_avg_pool2d(feat1, 1).flatten(1),
            F.adaptive_avg_pool2d(feat2, 1).flatten(1)
        ], dim=1)

        weights = self.attention(combined)  # [B, 2]
        w1, w2 = weights[:, 0:1, None, None], weights[:, 1:2, None, None]

        return w1 * feat1 + w2 * feat2
```

**장점:**
- 입력에 따라 동적으로 가중치 결정
- End-to-end 학습 가능
- 각 backbone의 강점을 상황에 맞게 활용

---

## 실험 우선순위

| 순위 | 방법 | 이유 |
|------|------|------|
| 1 | Ensemble Teacher | 구현 간단, 효과 검증됨 |
| 2 | Progressive Warmup | 기존 방식에 쉽게 추가 가능 |
| 3 | One-way KD | Inference 효율성 필요시 |
| 4 | Heatmap KL Div | 논문 방식 재현 |
| 5 | Correlation Alignment | Diversity 중요시 |
| 6 | Attention Fusion | 복잡하지만 유연함 |

---

## 참고 논문

1. **Deep Mutual Learning** (Zhang et al., 2017)
   - arXiv: 1706.00384
   - 핵심: 두 네트워크가 서로의 출력 분포를 모방

2. **Knowledge Distillation** (Hinton et al., 2015)
   - arXiv: 1503.02531
   - 핵심: Temperature scaling, soft targets

3. **CORAL (Correlation Alignment)** (Sun & Saenko, 2016)
   - 핵심: Feature correlation 기반 domain adaptation

---

## 현재 실험 결과 (Updated: 2026-01-20)

### Wandb 프로젝트별 결과

**mmpose_xregopose_single_coco:**
| Run | State | Full Body | Upper Body | Lower Body |
|-----|-------|-----------|------------|------------|
| fresh-haze-6 | running | **42.07mm** | 29.15mm | 55.00mm |

**mmpose_xregopose_coco_mpii:**
| Run | State | Full Body | Upper Body | Lower Body |
|-----|-------|-----------|------------|------------|
| full셋 test로 val | finished | 44.91mm | 30.56mm | 59.26mm |

### 핵심 비교

| Model | Full Body MPJPE | 차이 | 비고 |
|-------|-----------------|------|------|
| **Single (COCO)** | **42.07mm** | - | 🏆 현재 최고 |
| Dual (COCO+MPII) | 44.91mm | +2.84mm | mutual learning 적용 |

> **문제: Mutual learning이 오히려 성능을 2.84mm 악화시킴**
> 목표: Dual backbone mutual learning으로 Single (42mm) 이하 달성

---

## 구조적 문제: Heatmap의 3D 정보 인코딩 한계

### 핵심 문제

현재 구조에서 **Heatmap이 3D 정보를 인코딩해야 하는 부담**을 가짐:

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

1. **Depth Ambiguity 문제** ([A Survey on Depth Ambiguity](https://www.mdpi.com/2076-3417/12/20/10591))
   - "하나의 2D pose가 여러 3D pose로 매핑될 수 있음"
   - 2D heatmap은 본질적으로 **x, y 위치의 확률 분포**만 표현

2. **CNN Encoder의 정보 손실** ([EgoTAP, CVPR 2024](https://arxiv.org/html/2402.18330))
   - "CNN 기반 인코더가 heatmap 정보를 제대로 보존하지 못함"
   - 원거리 픽셀 간 통신 불가능 → spatial 관계 손실

3. **Lifting by Image** ([arXiv 2312.15636](https://arxiv.org/abs/2312.15636))
   - "2D pose만으로 lifting할 때 depth ambiguity 존재"
   - "이미지의 풍부한 semantic/texture 정보가 더 정확한 lifting에 기여"

4. **Multi-Feature Fusion** ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1047320324001159))
   - "2D prior + multi-scale features + backbone semantic features 융합이 효과적"

### 결론

**Heatmap은 2D 위치 인코딩에 최적화**되어 있으며, 3D depth 정보를 담기 어려움.
→ **Backbone feature를 별도로 활용**하여 depth cues 보존 필요.

---

## 제안: Backbone Feature Fusion

### 방안 A: Backbone Feature + Heatmap Latent Concatenation (추천)

```
Backbone feat [2048, 8, 8]
       │
       ├──→ GAP → [2048] → FC → [256]  ← Backbone latent (depth cues 보존)
       │                        │
       ↓ (Deconv)               │
Heatmap [16, 47, 47]            │
       ↓ (Encoder)              │
Z_hm [64]  ← 2D 위치 정보       │
       │                        │
       └──── Concat ────────────┘
              ↓
         [64 + 256] = [320]
              ↓
         Pose Decoder → 3D Pose
```

**구현:**
```python
class BackboneFusionHead(nn.Module):
    def __init__(self, backbone_channels=2048, heatmap_latent=64, backbone_latent=256):
        super().__init__()
        # Backbone feature → latent
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)
        self.backbone_fc = nn.Sequential(
            nn.Linear(backbone_channels, 512),
            nn.ReLU(),
            nn.Linear(512, backbone_latent)
        )

        # Heatmap encoder (기존)
        self.heatmap_encoder = Encoder(num_classes=16, output_size=heatmap_latent)

        # Pose decoder (입력 크기 증가)
        self.pose_decoder = LinearModel(
            input_size=heatmap_latent + backbone_latent + 64,  # Z_hm + Z_backbone + HMD
            num_classes=16
        )

    def forward(self, backbone_feat, heatmap, hmd_info):
        # Backbone latent (depth cues)
        z_backbone = self.backbone_pool(backbone_feat).flatten(1)  # [B, 2048]
        z_backbone = self.backbone_fc(z_backbone)                   # [B, 256]

        # Heatmap latent (2D position)
        z_heatmap = self.heatmap_encoder(heatmap)                   # [B, 64]

        # HMD info
        z_hmd = self.hmd_linear(hmd_info)                           # [B, 64]

        # Fusion
        z_fused = torch.cat([z_heatmap, z_backbone, z_hmd], dim=1)  # [B, 384]

        # 3D Pose
        pose_3d = self.pose_decoder(z_fused)
        return pose_3d
```

**역할 분리:**
| Component | 역할 | 정보 |
|-----------|------|------|
| Z_heatmap | 정확한 2D 관절 위치 | x, y coordinates |
| Z_backbone | 깊이/텍스처/컨텍스트 | depth cues, scene context |
| Z_hmd | 머리/손 3D 위치 | absolute 3D reference |

**장점:**
- Heatmap: 2D 위치 정보에 집중
- Backbone: depth/context 정보 보존
- 각각 최적화 가능 (역할 분리)

---

### 방안 B: 2D Coords + Backbone Feature ([Lifting by Image](https://arxiv.org/abs/2312.15636) 방식)

```
Heatmap → soft-argmax → 2D coords [16, 2] + conf [16]
                              │
Backbone feat → GAP → FC → Context [256]
                              │
              Concat ─────────┘
                 ↓
           [32 + 16 + 256 + 9] = [313]
                 ↓
           Lifting Network → 3D Pose
```

**장점:**
- 2D 좌표가 명시적 (해석 가능)
- Backbone feature가 depth ambiguity 해결에 직접 기여
- Martinez baseline과 호환

---

### 방안 C: Multi-scale Backbone Feature Fusion

```
Backbone feat [2048, 8, 8]
       │
       ├── Stage 4: [2048, 8, 8] → GAP → [256]
       │
Backbone feat [1024, 16, 16]  (from earlier stage)
       │
       ├── Stage 3: [1024, 16, 16] → GAP → [128]
       │
       └──── Concat → [384] ← Multi-scale context
                 │
                 + Heatmap latent [64] + HMD [64]
                 │
           Pose Decoder → 3D Pose
```

**장점:**
- 다양한 해상도의 정보 활용
- 세밀한 정보 + 전역 정보 조합

---

### 방안 비교

| 방안 | 구현 난이도 | 정보 분리 | Depth 정보 | 추천 |
|------|------------|----------|-----------|------|
| 현재 (Heatmap only) | - | ✗ | ✗ | - |
| **A: Backbone + Heatmap Concat** | **낮음** | **✓✓** | **✓✓** | **✓✓ (먼저)** |
| B: 2D Coords + Backbone | 중간 | ✓✓✓ | ✓✓ | ✓✓ |
| C: Multi-scale Fusion | 높음 | ✓✓ | ✓✓✓ | ✓ |

**추천 순서:**
1. **방안 A 먼저 시도** - 최소 변경으로 효과 검증
2. 효과 있으면 방안 B로 확장 (2D 좌표 명시화)
3. 필요시 방안 C (multi-scale)

---

## 제안: Lifting + Backbone Feature Fusion (구현 예정)

### 배경

**구현 완료된 Lifting Head** (`CustomEgoposeLiftingHead`):
```
Backbone → Heatmap → soft_argmax → 2D coords [16,2] + conf [16]
                                        ↓
                              [2D: 32] + [conf: 16] + [HMD: 9] = 57
                                        ↓
                              Lifting Network (4M params)
                                        ↓
                                   3D Pose
```

**문제**: 여전히 **Backbone feature의 depth cues를 활용하지 못함**
- soft_argmax는 2D 위치만 추출 (depth 정보 없음)
- Lifting network 입력에 spatial/context 정보 부족

### 제안 구조: Lifting + Backbone Fusion

```
Backbone feat [2048, 8, 8]
       │
       ├──────────────────────────────────┐
       │                                  │
       ↓ (Deconv)                         ↓ (GAP → FC)
Heatmap [16, 47, 47]                Z_backbone [256]
       │                                  │
       ↓ (soft_argmax)                    │ (depth/context cues)
2D coords [16,2] + conf [16]              │
       │                                  │
       └────────── Concat ────────────────┘
                     ↓
         [32 + 16 + 256 + 9(HMD)] = 313
                     ↓
              Lifting Network
                     ↓
                3D Pose [16, 3]
```

### 역할 분리 (명확!)

| Component | 차원 | 역할 | 정보 유형 |
|-----------|------|------|----------|
| 2D coords | [16, 2] = 32 | 정확한 관절 위치 | x, y (sub-pixel) |
| confidence | [16] | 2D 예측 신뢰도 | 가려짐/불확실성 |
| **Z_backbone** | **[256]** | **depth/texture/context** | **3D 단서** |
| HMD info | [9] | 머리/손 3D 위치 | absolute reference |

### 핵심 가설

> **Backbone feature는 3D 정보를 implicit하게 인코딩하고 있다**
>
> - Texture gradients → depth cues
> - Scene context → scale/distance 추정
> - Body part relationships → 3D structure
>
> Heatmap은 2D 위치에 집중, Backbone은 3D context 제공 → **상호 보완**

### 학술적 근거

1. **[Lifting by Image](https://arxiv.org/abs/2312.15636)**:
   - "이미지의 semantic/texture 정보가 더 정확한 3D lifting에 기여"
   - 2D pose만으로는 depth ambiguity 해결 불가

2. **[Multi-Feature Fusion](https://www.sciencedirect.com/science/article/abs/pii/S1047320324001159)**:
   - "2D prior + backbone semantic features 융합이 효과적"

3. **Depth from single image 연구들**:
   - CNN은 texture gradients, occlusion 등에서 depth 추론 가능
   - Backbone feature에 이미 depth cues 존재

### 구현 계획

**파일**: `custom_egopose_lifting_backbone_fusion_head.py`

```python
class LiftingBackboneFusionHead(CustomEgoposeLiftingHead):
    def __init__(self, ..., backbone_latent_dim=256):
        super().__init__(...)

        # Backbone feature → latent
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)
        self.backbone_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, backbone_latent_dim)
        )

        # Lifting network 입력 크기 증가
        # 기존: 32 + 16 + 9 = 57
        # 변경: 32 + 16 + 256 + 9 = 313
        self.lifting_network = LiftingNetwork(
            input_dim=32 + 16 + backbone_latent_dim + 9,
            ...
        )

    def forward_lifting(self, heatmaps, backbone_feat, hmd_info):
        # 2D coords from heatmap
        coords_2d, confidence = soft_argmax_2d(heatmaps)

        # Backbone latent (depth cues)
        z_backbone = self.backbone_pool(backbone_feat).flatten(1)
        z_backbone = self.backbone_fc(z_backbone)

        # Lifting with all features
        pose_3d = self.lifting_network(coords_2d, confidence, z_backbone, hmd_info)

        return pose_3d, coords_2d, confidence
```

### 실험 계획

| 실험 | 구성 | 비교 대상 |
|------|------|----------|
| Lifting (현재) | 2D + conf + HMD | Single COCO (42.07mm) |
| **Lifting + Backbone** | 2D + conf + **Z_backbone** + HMD | Lifting 결과 |

**성공 기준**: Lifting 대비 MPJPE 개선

### 파라미터 비교

| 모델 | Head Params | 입력 차원 |
|------|-------------|----------|
| 기존 (Encoder/Decoder) | ~61M | Z[64] + HMD[64] |
| Lifting (현재) | ~13M | 2D[32] + conf[16] + HMD[9] = 57 |
| **Lifting + Backbone** | **~14M** | **2D[32] + conf[16] + backbone[256] + HMD[9] = 313** |

> Backbone fusion 추가해도 파라미터 증가 미미 (~1M)
> 하지만 depth cues로 인한 성능 향상 기대

---

## 구조적 문제: Heatmap → Latent 압축

### 현재 파이프라인

```
Backbone Feature [2048, 8, 8]     ← 풍부한 spatial + semantic 정보
         ↓ (Deconv)
    Heatmap [16, 47, 47]          ← 관절 위치 확률 분포
         ↓ (Encoder - Conv layers)
      Z [64]                      ← 극단적 압축
         ↓ + HMD info [9→64]
    Pose Decoder
         ↓
    3D Pose [16, 3]
```

### 문제점

1. **이중 압축으로 인한 정보 손실**
   - Heatmap: 35,344 values (16×47×47)
   - Latent Z: 64 values
   - **99.8% 정보 손실** - 공간적 관계, 관절 간 상대 위치 정보 손실

2. **Heatmap 목적과 불일치**
   - Heatmap 원래 용도: argmax/soft-argmax로 2D 좌표 추출
   - 현재 사용: 통째로 압축 → 목적에 맞지 않음

3. **불필요한 bottleneck**
   - Backbone feature가 이미 풍부한 정보 보유
   - Heatmap 단계에서 2D 위치 정보로 제한
   - 다시 압축하면서 추가 손실

### 대안 1: Backbone Feature 직접 사용 (추천)

```
Backbone Feature [2048, 8, 8]
         ↓ (Global Average Pool)
    Feature [2048]
         ↓ (FC layers)
      Z [256~512]
         ↓ + HMD info
    3D Pose [16, 3]
```

```python
class DirectFeatureHead(nn.Module):
    def __init__(self, in_channels=2048, latent_dim=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.pose_decoder = LinearModel(
            input_size=latent_dim + 64,  # + HMD info
            num_classes=16
        )

    def forward(self, backbone_feat, hmd_info):
        x = self.pool(backbone_feat).flatten(1)  # [B, 2048]
        z = self.fc(x)                            # [B, 256]
        hmd = self.hmd_linear(hmd_info)           # [B, 64]
        pose_3d = self.pose_decoder(torch.cat([z, hmd], dim=1))
        return pose_3d
```

**장점:**
- Heatmap bottleneck 제거
- 더 풍부한 정보 유지
- 연산량 감소

---

### 대안 2: 2D Keypoints → 3D Lifting (Martinez Baseline)

```
Heatmap [16, 47, 47]
         ↓ (soft-argmax)
  2D Keypoints [16, 2]  + Confidence [16]
         ↓
    [16 × 3 = 48] + HMD [9]
         ↓ (Linear layers)
    3D Pose [16, 3]
```

```python
def soft_argmax(heatmaps):
    """Heatmap에서 미분 가능한 2D 좌표 추출"""
    B, K, H, W = heatmaps.shape

    # Softmax over spatial dimensions
    heatmaps_flat = heatmaps.view(B, K, -1)
    heatmaps_soft = F.softmax(heatmaps_flat, dim=-1)

    # Create coordinate grids
    x_coords = torch.linspace(0, 1, W).to(heatmaps.device)
    y_coords = torch.linspace(0, 1, H).to(heatmaps.device)

    # Compute expected coordinates
    x = (heatmaps_soft * x_coords.view(1, 1, -1).expand(B, K, H*W)).sum(dim=-1)
    y = (heatmaps_soft * y_coords.view(1, 1, -1).expand(B, K, H*W)).sum(dim=-1)

    # Confidence = max value
    confidence = heatmaps.view(B, K, -1).max(dim=-1)[0]

    return torch.stack([x, y], dim=-1), confidence  # [B, 16, 2], [B, 16]

class Lifting2Dto3D(nn.Module):
    def __init__(self, num_joints=16):
        super().__init__()
        # Input: 2D coords (16*2) + confidence (16) + HMD (9) = 57
        self.fc = nn.Sequential(
            nn.Linear(57, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_joints * 3)
        )

    def forward(self, heatmaps, hmd_info):
        coords_2d, conf = soft_argmax(heatmaps)  # [B, 16, 2], [B, 16]
        x = torch.cat([
            coords_2d.flatten(1),  # [B, 32]
            conf,                   # [B, 16]
            hmd_info                # [B, 9]
        ], dim=1)
        return self.fc(x).view(-1, 16, 3)
```

**장점:**
- 검증된 baseline 방식 (Martinez et al., 2017)
- 2D 정확도와 3D 분리 가능
- 해석 가능한 중간 표현

---

### 대안 3: Heatmap Spatial Feature 유지

```
Heatmap [16, 47, 47]
         ↓ (Conv layers - spatial 유지)
   Feature [64, 12, 12]
         ↓ (flatten)
      [9216]
         ↓ + HMD info
    3D Pose [16, 3]
```

```python
class SpatialHeatmapEncoder(nn.Module):
    def __init__(self, num_joints=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_joints, 32, 3, stride=2, padding=1),  # 47→24
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),          # 24→12
            nn.ReLU(),
        )
        # 64 * 12 * 12 = 9216
        self.fc = nn.Linear(9216 + 64, 256)  # + HMD

    def forward(self, heatmap, hmd_info):
        x = self.conv(heatmap)           # [B, 64, 12, 12]
        x = x.flatten(1)                  # [B, 9216]
        hmd = self.hmd_linear(hmd_info)   # [B, 64]
        z = self.fc(torch.cat([x, hmd], dim=1))
        return z
```

**장점:**
- Spatial 정보 부분적 유지
- 현재 구조에서 최소 변경

---

### 대안 비교

| 방식 | 정보 보존 | 구현 난이도 | 권장 |
|------|----------|------------|------|
| 현재 (Heatmap→Z 64) | ★☆☆☆☆ | - | ✗ |
| Backbone Feature 직접 | ★★★★★ | 중 | ✓✓ |
| 2D→3D Lifting | ★★★☆☆ | 하 | ✓✓ |
| Heatmap Spatial 유지 | ★★★☆☆ | 하 | ✓ |

**권장 순서:**
1. **Backbone Feature 직접 사용** - 가장 많은 정보 유지
2. **2D→3D Lifting** - 검증된 방식, 해석 가능
3. **Heatmap Spatial 유지** - 최소 변경으로 개선

---

## 현재 훈련 모델 구조 (COCO+MPII)

```python
# Config: HMD_xregopose_h5cache_coco_mpii_config.py
model = dict(
    type='Custom_TopdownPoseEstimator',
    backbone=dict(type='ResNet', depth=101, init_cfg=COCO_pretrained),
    backbone2=dict(type='ResNet', depth=101, init_cfg=MPII_pretrained),
    head=dict(
        type='CustomxRegoposeBaselinel1_multi_backbone',
        in_channels=2048,
        out_channels=16,
        # Losses
        loss=dict(type='KeypointMSELoss', loss_weight=1000),        # 2D heatmap
        loss_heatmap_recon=dict(loss_weight=250),                   # Heatmap reconstruction
        loss_pose_l2norm=dict(loss_weight=1.0),                     # 3D L2
        loss_cosine_similarity=dict(loss_weight=0.1),               # Cosine sim
        loss_limb_length=dict(loss_weight=0.25),                    # Limb length
        loss_backbone_latant=dict(loss_weight=1.0),                 # Backbone MSE
        loss_backbone_heatmap=dict(loss_weight=1.0),                # Sub heatmap
    )
)
```

**Loss 구성:**
- `loss_kpt` (w=1000): Main heatmap supervision
- `loss_heatmap_recon` (w=250): Z에서 heatmap 복원
- `loss_pose_l2norm` (w=1.0): 3D pose L2 loss
- `loss_backbone_latant` (w=1.0): Dual backbone feature MSE
- `loss_backbone_heatmap` (w=1.0): Sub backbone heatmap loss

---

## HMD 정보 연결 위치 분석

### 현재 구조

```
Backbone → Heatmap [16,47,47] → Encoder → Z[64]
                                            ↓
                                      Z + HMD[64]  ← 현재 위치
                                            ↓
                                      Pose Decoder → 3D Pose
```

### 옵션 1: 현재 위치 (Pose Decoder 전)

| 장점 | 단점 |
|------|------|
| 3D 예측 직전에 3D 정보 주입 | Heatmap bottleneck 이후라 정보 이미 손실 |
| 구현 간단 | HMD가 heatmap 품질 개선에 기여 못함 |

### 옵션 2: Heatmap 생성 전 (추천)

```
Backbone feat + HMD → Deconv → Heatmap
                                 ↓
                          (가려진 관절도 heatmap에 반영)
```

```python
class HMDConditionedHeatmapHead(nn.Module):
    def __init__(self):
        self.hmd_embed = nn.Linear(9, 256)

    def forward(self, backbone_feat, hmd_info):
        hmd_feat = self.hmd_embed(hmd_info)  # [B, 256]
        hmd_spatial = hmd_feat[:, :, None, None].expand(-1, -1, H, W)

        # Backbone feature와 결합
        fused = torch.cat([backbone_feat, hmd_spatial], dim=1)
        heatmap = self.deconv(fused)
        return heatmap
```

| 장점 | 단점 |
|------|------|
| HMD가 heatmap 품질 개선 가능 | 구조 변경 필요 |
| 가려진 관절 위치 힌트 제공 | HMD 의존도 높아질 수 있음 |

### 옵션 3: 2D→3D Lifting (가장 깔끔)

```
Heatmap → soft-argmax → 2D coords[16,2] + conf[16]
                                ↓
                    [2D: 32] + [conf: 16] + [HMD: 9] = 57
                                ↓
                          Lifting Network → 3D Pose
```

| 장점 | 단점 |
|------|------|
| Martinez baseline 방식 (검증됨) | Heatmap spatial 정보 일부 손실 |
| 해석 가능한 구조 | |
| HMD가 depth ambiguity 해결에 직접 기여 | |

### HMD 연결 위치 추천

| 순위 | 방식 | 이유 |
|------|------|------|
| 1 | 2D→3D Lifting | 가장 깔끔, HMD가 depth 추정에 직접 기여 |
| 2 | Heatmap 생성 전 | 가려진 관절 heatmap 품질 개선 |
| 3 | 현재 유지 + Z 크기↑ | 최소 변경으로 개선 |

---

## Cross Attention 기반 HMD Fusion

### 방안 1: HMD가 Visual Feature에 Attend

```
Visual Feature (from Backbone/Heatmap)
        ↓
   ┌─────────────────────────────────┐
   │      Cross Attention            │
   │  Q: HMD                         │
   │  K, V: Visual Feature           │
   │                                 │
   │  "HMD 관점에서 중요한 visual    │
   │   정보만 선택적으로 추출"        │
   └─────────────────────────────────┘
        ↓
   HMD-aware Feature → 3D Pose
```

```python
class HMDCrossAttention(nn.Module):
    def __init__(self, visual_dim=256, hmd_dim=9, hidden_dim=64, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scale = hidden_dim ** -0.5

        # HMD를 query로
        self.hmd_to_query = nn.Linear(hmd_dim, hidden_dim * num_heads)
        # Visual feature를 key, value로
        self.visual_to_key = nn.Linear(visual_dim, hidden_dim * num_heads)
        self.visual_to_value = nn.Linear(visual_dim, hidden_dim * num_heads)

        self.out_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, visual_feat, hmd_info):
        """
        visual_feat: [B, N, C] (N = H*W spatial tokens)
        hmd_info: [B, 9]
        """
        B = visual_feat.size(0)

        Q = self.hmd_to_query(hmd_info).unsqueeze(1)  # [B, 1, heads*dim]
        K = self.visual_to_key(visual_feat)           # [B, N, heads*dim]
        V = self.visual_to_value(visual_feat)

        # Multi-head reshape
        Q = Q.view(B, 1, self.num_heads, self.hidden_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.hidden_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.hidden_dim).transpose(1, 2)

        # Attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ V).transpose(1, 2).reshape(B, 1, -1)
        return self.out_proj(out.squeeze(1))  # [B, hidden_dim]
```

### 방안 2: Joint-wise Cross Attention (추천)

각 관절이 독립적으로 HMD 정보에 attend:

```
Heatmap [16, 47, 47]
        ↓
   각 관절별 feature 추출 (16개 token)
        ↓
   ┌─────────────────────────────────┐
   │      Cross Attention            │
   │  Q: Joint tokens [16, D]        │
   │  K, V: HMD tokens [3, D]        │
   │       (head, right_hand, left)  │
   │                                 │
   │  "각 관절이 관련 HMD 정보에     │
   │   선택적으로 attend"            │
   └─────────────────────────────────┘
        ↓
   HMD-aware Joint Features → 3D Pose
```

```python
class JointHMDCrossAttention(nn.Module):
    def __init__(self, joint_dim=64, num_joints=16, num_heads=4):
        super().__init__()

        # HMD를 3개 token으로 (head, right_hand, left_hand)
        self.hmd_embed = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * joint_dim)  # 3 tokens
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(joint_dim)

    def forward(self, joint_features, hmd_info):
        """
        joint_features: [B, 16, D] (각 관절의 feature)
        hmd_info: [B, 9]
        """
        B = joint_features.size(0)

        # HMD → 3개 token [B, 3, D]
        hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)

        # Cross attention: joints attend to HMD
        attn_out, _ = self.cross_attn(
            query=joint_features,   # [B, 16, D]
            key=hmd_tokens,         # [B, 3, D]
            value=hmd_tokens        # [B, 3, D]
        )

        # Residual connection
        out = self.norm(joint_features + attn_out)
        return out  # [B, 16, D]
```

**장점:**
- 손 관절 → 손 HMD에 attend
- 몸통 관절 → head HMD에 attend
- 자연스러운 semantic 매핑

### 방안 3: Bi-directional Cross Attention

양방향 정보 교환:

```
Visual Feature ←→ HMD Feature
     ↓                ↓
  V attends        HMD attends
  to HMD           to Visual
     ↓                ↓
     └──── Fusion ────┘
              ↓
          3D Pose
```

```python
class BidirectionalCrossAttention(nn.Module):
    def __init__(self, visual_dim=256, hmd_dim=64, num_heads=4):
        super().__init__()

        self.visual_to_hmd = nn.MultiheadAttention(
            visual_dim, num_heads, batch_first=True)
        self.hmd_to_visual = nn.MultiheadAttention(
            hmd_dim, num_heads, batch_first=True)

        self.hmd_expand = nn.Linear(9, hmd_dim)
        self.visual_proj = nn.Linear(visual_dim, hmd_dim)
        self.fusion = nn.Linear(visual_dim + hmd_dim, 256)

    def forward(self, visual_tokens, hmd_info):
        hmd_feat = self.hmd_expand(hmd_info).unsqueeze(1)  # [B, 1, hmd_dim]

        # Visual attends to HMD
        v2h, _ = self.visual_to_hmd(
            query=visual_tokens, key=hmd_feat, value=hmd_feat)

        # HMD attends to Visual
        visual_proj = self.visual_proj(visual_tokens)
        h2v, _ = self.hmd_to_visual(
            query=hmd_feat, key=visual_proj, value=visual_proj)

        # Fusion
        v_enhanced = visual_tokens + v2h
        h_enhanced = h2v.mean(dim=1)
        v_global = v_enhanced.mean(dim=1)

        return self.fusion(torch.cat([v_global, h_enhanced], dim=-1))
```

### 전체 파이프라인 제안 (Multi-stage Cross Attention)

```
Backbone Feature [B, 2048, 8, 8]
        ↓
   Flatten → [B, 64, 2048]  (spatial tokens)
        ↓
   ┌─────────────────────────────────┐
   │   Cross Attention Block 1       │
   │   Visual ← HMD                  │
   │   (HMD로 중요 영역 강조)         │
   └─────────────────────────────────┘
        ↓
   Deconv → Heatmap [B, 16, 47, 47]
        ↓
   Soft-argmax → Joint tokens [B, 16, D]
        ↓
   ┌─────────────────────────────────┐
   │   Cross Attention Block 2       │
   │   Joints ← HMD                  │
   │   (각 관절이 HMD 정보 참조)      │
   └─────────────────────────────────┘
        ↓
   Pose Decoder → 3D Pose [B, 16, 3]
```

### Cross Attention 방식 비교

| 방식 | 복잡도 | 효과 | 추천 |
|------|--------|------|------|
| 단순 concat (현재) | 낮음 | 보통 | - |
| HMD→Visual attention | 중간 | 좋음 | ✓ |
| **Joint-wise attention** | 중간 | **매우 좋음** | **✓✓** |
| Bi-directional | 높음 | 좋음 | ✓ |
| Multi-stage | 높음 | 매우 좋음 | ✓ |

**Joint-wise Cross Attention 추천 이유:**
- 각 관절이 관련 HMD 정보만 선택적 사용
- 손 관절 → 손 HMD, 몸통 → head HMD로 자연스러운 매핑
- Transformer 구조와 호환 (확장 용이)

---

## Head 구조 최적화: HeatmapDecoder 파라미터 문제

### 현재 문제

Head 전체 파라미터: **61.4M** (backbone 42.5M보다 큼!)

| Component | Params | 비고 |
|-----------|--------|------|
| **heatmap_decoder** | **40.04M** | ← 문제! |
| deconv_layers | 9.44M | backbone1용 |
| deconv_layers_ | 9.44M | backbone2용 |
| encoder | 0.71M | |
| pose_decoder | 0.59M | |

### 원인 분석

`heatmap_decoder` 내부 구조:
```python
HeatmapDecoder(
  (linear1): Linear(64 → 512)       # 0.03M
  (linear2): Linear(512 → 2048)     # 1.05M
  (linear3): Linear(2048 → 18432)   # 37.77M  ← 범인!
  (deconv1): ConvTranspose2d(...)   # 1.05M
  (deconv2): ConvTranspose2d(...)   # 0.13M
  (deconv3): ConvTranspose2d(...)   # ...
)
```

**`linear3`의 문제:**
- 입력: 2048 (latent vector)
- 출력: 18432 = 512 × 6 × 6 (reshape to spatial)
- 파라미터: 2048 × 18432 = **37.77M**

FC layer로 spatial feature를 생성하는 것이 비효율적.

### 해결 방안 1: 1×1 Conv + Upsample (추천)

```python
class EfficientHeatmapDecoder(nn.Module):
    """FC 대신 Conv로 spatial 생성 - 파라미터 95% 감소"""
    def __init__(self, latent_dim=64, out_channels=16):
        super().__init__()

        # Latent → small spatial (1×1 → 6×6)
        self.fc = nn.Linear(latent_dim, 256)  # 64 → 256 (0.02M)

        # Reshape to 256×1×1, then upsample
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 1→2  (0.13M)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 2→4  (0.13M)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),    # 4→8  (0.07M)
            nn.ReLU(),
        )

        # 8×8 → 47×47
        self.to_heatmap = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),   # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 0),  # 32→47
        )

    def forward(self, z):
        x = self.fc(z)                    # [B, 256]
        x = x.view(-1, 256, 1, 1)         # [B, 256, 1, 1]
        x = self.upsample(x)              # [B, 64, 8, 8]
        heatmap = self.to_heatmap(x)      # [B, 16, 47, 47]
        return heatmap
```

**파라미터 비교:**
- 현재: 40.04M
- 개선: ~1.5M (**96% 감소**)

### 해결 방안 2: AdaIN 기반 생성

```python
class AdaINHeatmapDecoder(nn.Module):
    """Latent를 style로 사용, learned constant에서 시작"""
    def __init__(self, latent_dim=64, out_channels=16):
        super().__init__()

        # Learned constant (spatial starting point)
        self.const = nn.Parameter(torch.randn(1, 256, 6, 6))

        # Latent → AdaIN parameters
        self.style_fc = nn.Linear(latent_dim, 256 * 2)  # scale + shift

        # Upsampling with AdaIN
        self.conv_blocks = nn.ModuleList([
            nn.Conv2d(256, 128, 3, padding=1),  # 6×6
            nn.Conv2d(128, 64, 3, padding=1),   # 12×12
            nn.Conv2d(64, out_channels, 3, padding=1),  # 24×24 → 47
        ])

    def adain(self, feat, style):
        scale, shift = style.chunk(2, dim=1)
        return feat * scale[:,:,None,None] + shift[:,:,None,None]

    def forward(self, z):
        style = self.style_fc(z)
        x = self.const.expand(z.size(0), -1, -1, -1)
        x = self.adain(x, style)
        # ... upsample
        return heatmap
```

### 해결 방안 3: Heatmap Decoder 제거

2D→3D Lifting 방식에서는 heatmap_decoder 자체가 불필요:

```python
# 현재: Heatmap → Encoder → Z → HeatmapDecoder → ReconHeatmap
# 제안: Heatmap → soft_argmax → 2D coords → Lifting → 3D
```

이 경우 heatmap_decoder 40M 전체 제거 가능.

### 메모리 영향

| 구성 | Head Params | 전체 Params | batch=58 메모리 |
|------|-------------|-------------|-----------------|
| 현재 | 61.4M | 146.4M | ~22GB (OOM 위험) |
| Conv 기반 | ~22M | ~107M | ~16GB (안전) |
| Decoder 제거 | ~12M | ~97M | ~14GB (매우 안전) |

### 권장 순서

1. **단기 (즉시)**: batch_size=48로 줄여서 훈련 진행
2. **중기**: Conv 기반 decoder로 교체 (파라미터 96% 감소)
3. **장기**: 2D→3D Lifting으로 전환 (decoder 완전 제거)
