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

## 현재 실험 결과

| Model | Full Body MPJPE | 비고 |
|-------|-----------------|------|
| Dual (MPII+MPII) | 96.47mm | 같은 pretrained |
| Single (COCO) | **77.27mm** | 단일 backbone |
| Dual (COCO+MPII) | 진행중 | 다른 pretrained |

> Single COCO가 현재 최고 성능. Dual backbone의 개선이 필요함.
