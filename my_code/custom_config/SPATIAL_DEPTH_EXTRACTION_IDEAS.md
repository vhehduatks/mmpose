# 공간 정보 보존 Depth 추출 방법

> 생성일: 2026-01-21

## 문제 정의

### 현재 Phase 5-C 구조의 한계

```
Backbone [2048, 8, 8]
       │
       ├→ Deconv → Heatmap → soft_argmax → coords_2d (detach)
       │                                        │
       └→ GAP → FC → Z_backbone [256] ──────────┼→ Lifting → 3D
              ❌ 공간 정보 손실!                 │
                                                └→ 3D Loss
```

**문제점**:
- Global Average Pooling이 모든 공간 정보를 평균으로 압축
- "왼쪽 팔꿈치의 depth"를 학습하려면 **어디가 왼쪽 팔꿈치인지** 알아야 함
- Z_backbone [256]은 전체 이미지의 global context만 포함
- 결과: MPJPE 176mm (baseline 41mm 대비 4배 이상 나쁨)

### 목표

```
각 관절의 2D 위치에서 해당 위치의 backbone feature를 추출
→ 관절별 depth cue 학습 가능
→ 2D/3D 역할 분리 유지하면서 공간 정보 보존
```

---

## 방법 비교

| 방법 | 공간 정보 | 구현 복잡도 | 메모리 | 특징 |
|------|----------|------------|--------|------|
| GAP (현재) | ❌ 손실 | 낮음 | 낮음 | 실패 |
| **1. Grid Sampling** | ✅ 정확 | **낮음** | 낮음 | **권장** |
| 2. Heatmap Weighted | ✅ soft | 낮음 | 중간 | Soft attention |
| 3. Cross-Attention | ✅ 학습됨 | 높음 | 높음 | 유연함 |
| 4. Deformable Attn | ✅ 학습됨 | 높음 | 중간 | Sparse sampling |

---

## 방법 1: Grid Sampling (권장)

### 핵심 아이디어

2D 좌표를 사용하여 backbone feature map에서 해당 위치의 feature를 직접 샘플링.

```
coords_2d [B, 16, 2]  →  backbone_feat [B, 2048, 8, 8]
         │                        │
         └────── grid_sample ─────┘
                      ↓
            joint_features [B, 16, 2048]
```

### 구조도

```
Backbone feat [B, 2048, 8, 8]
       │
       ├─────────────────────────────────────┐
       │                                     │
       ↓ (Deconv)                            │
Heatmap [B, 16, 47, 47]                      │
       │                                     │
       ↓ (soft_argmax)                       │
coords_2d [B, 16, 2]                         │
       │                                     │
       │ (detach)                            │
       │                                     │
       ↓                                     ↓
coords_2d_detached ──────────────→ grid_sample(backbone, coords)
                                             │
                                             ↓
                                  joint_features [B, 16, 2048]
                                             │
                                             ↓ (FC)
                                  joint_depth [B, 16, 64]
                                             │
       ┌─────────────────────────────────────┘
       │
       ↓
Concat [coords_2d_detached, confidence, joint_depth, hmd_info]
       │
       ↓
Lifting Network → 3D Pose
```

### 상세 구현

```python
import torch
import torch.nn.functional as F

class SpatialDepthExtractor(nn.Module):
    """
    2D 좌표 위치에서 backbone feature 추출
    """
    def __init__(self, in_channels: int = 2048, out_channels: int = 64):
        super().__init__()
        # 관절별 feature를 압축
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, backbone_feat: Tensor, coords_2d: Tensor) -> Tensor:
        """
        Args:
            backbone_feat: [B, C, H, W] backbone feature map
            coords_2d: [B, K, 2] normalized coordinates (0~1)

        Returns:
            joint_features: [B, K, out_channels] 관절별 depth feature
        """
        B, K, _ = coords_2d.shape

        # Step 1: coords를 grid_sample 형식으로 변환
        # grid_sample은 [-1, 1] 범위 사용
        coords_normalized = coords_2d * 2 - 1  # [0,1] → [-1,1]

        # Step 2: grid 형태로 reshape
        # grid_sample expects: [B, H_out, W_out, 2]
        # 우리는 K개 점을 샘플링: [B, K, 1, 2]
        coords_grid = coords_normalized.unsqueeze(2)  # [B, K, 1, 2]

        # Step 3: bilinear interpolation으로 feature 샘플링
        # backbone_feat: [B, C, H, W]
        # coords_grid: [B, K, 1, 2]
        # output: [B, C, K, 1]
        sampled = F.grid_sample(
            backbone_feat,
            coords_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        # Step 4: reshape to [B, K, C]
        sampled = sampled.squeeze(-1)  # [B, C, K]
        sampled = sampled.permute(0, 2, 1)  # [B, K, C]

        # Step 5: FC로 차원 축소
        joint_features = self.fc(sampled)  # [B, K, out_channels]

        return joint_features
```

### Gradient 흐름 분석

```
3D Loss
   │
   ↓
Lifting Network
   │
   ├─── coords_2d_detached ←── coords_2d ←── heatmap
   │         ❌ gradient 차단          ✅ heatmap loss로만 학습
   │
   └─── joint_depth
              │
              ↓ (FC)
        joint_features
              │
              ↓ (grid_sample)
        backbone_feat ←── Backbone
              ✅ 3D gradient 흐름!
```

**핵심**:
- coords_2d는 detach → heatmap은 2D loss로만 학습
- grid_sample은 미분 가능 → 3D loss가 backbone으로 흐름
- 각 관절 위치의 feature가 해당 관절의 depth 학습에 사용됨

### 좌표계 변환 상세

```python
# soft_argmax 출력: [0, 1] 범위 (normalized)
coords_2d = soft_argmax(heatmap)  # [B, K, 2], 범위 [0, 1]

# grid_sample 입력: [-1, 1] 범위
#   (-1, -1) = 좌상단
#   (+1, +1) = 우하단
coords_grid = coords_2d * 2 - 1  # [0,1] → [-1,1]

# 예시:
#   coords_2d = (0.0, 0.0) → coords_grid = (-1, -1) → 좌상단
#   coords_2d = (0.5, 0.5) → coords_grid = (0, 0)   → 중앙
#   coords_2d = (1.0, 1.0) → coords_grid = (1, 1)   → 우하단
```

### Backbone Feature Map 해상도

```
Backbone output: [B, 2048, 8, 8]
Heatmap: [B, 16, 47, 47]

문제: 해상도 불일치
- coords_2d는 47x47 heatmap 기준
- backbone은 8x8

해결: grid_sample의 bilinear interpolation
- coords_2d [0,1]을 8x8에 매핑
- 정수가 아닌 위치도 보간으로 처리
- 예: (0.3, 0.7) → 8x8에서 (2.4, 5.6) → 주변 4픽셀 보간
```

### 수정된 Lifting Network

```python
class LiftingNetworkWithSpatialDepth(nn.Module):
    def __init__(self,
                 num_joints: int = 16,
                 hmd_dim: int = 9,
                 depth_dim: int = 64,  # joint_depth 차원
                 hidden_dim: int = 1024):
        super().__init__()

        # Input: coords(32) + conf(16) + joint_depth(16*64) + HMD(9)
        # = 32 + 16 + 1024 + 9 = 1081
        # 또는 joint_depth를 flatten하지 않고 관절별 처리

        # Option A: Flatten all
        input_dim = num_joints * 2 + num_joints + num_joints * depth_dim + hmd_dim

        # Option B: Per-joint processing (권장)
        # 각 관절: coords(2) + conf(1) + depth(64) = 67
        # → 관절별 MLP → concat → final MLP
```

### 예상 장점

1. **공간 정보 보존**: 각 관절이 자신의 위치에서 feature 추출
2. **관절별 depth 학습**: 16개 관절 각각 독립적인 depth cue
3. **Gradient 분리 유지**: coords_2d는 여전히 detach
4. **구현 간단**: grid_sample 하나로 해결

### 예상 문제점 및 해결

| 문제 | 해결책 |
|------|--------|
| 8x8 해상도가 너무 낮음 | 중간 feature (16x16, 32x32) 사용 |
| 단일 점 샘플링 불안정 | 3x3 영역 pooling 또는 multi-scale |
| coords_2d 오차 전파 | temperature 조절로 sharp heatmap |

---

## 방법 2: Heatmap-Weighted Pooling

### 아이디어

Heatmap 자체를 spatial attention으로 사용하여 backbone feature의 weighted sum 계산.

```python
def heatmap_weighted_pooling(backbone_feat, heatmaps):
    """
    Args:
        backbone_feat: [B, C, H, W] - e.g., [B, 2048, 8, 8]
        heatmaps: [B, K, H', W'] - e.g., [B, 16, 47, 47]

    Returns:
        joint_features: [B, K, C]
    """
    B, C, H, W = backbone_feat.shape
    B, K, H2, W2 = heatmaps.shape

    # Resize heatmaps to backbone resolution
    heatmaps_resized = F.interpolate(
        heatmaps, size=(H, W), mode='bilinear', align_corners=True
    )  # [B, K, H, W]

    # Normalize to attention weights
    heatmaps_flat = heatmaps_resized.view(B, K, -1)  # [B, K, H*W]
    attn_weights = F.softmax(heatmaps_flat, dim=-1)   # [B, K, H*W]

    # Weighted sum
    backbone_flat = backbone_feat.view(B, C, -1)  # [B, C, H*W]

    # einsum: 관절별 weighted sum
    joint_features = torch.einsum('bkn,bcn->bkc', attn_weights, backbone_flat)
    # [B, K, C]

    return joint_features
```

### 장단점

**장점**:
- Soft attention으로 더 넓은 영역 참조
- Heatmap uncertainty가 자연스럽게 반영됨

**단점**:
- Heatmap이 detach되면 attention도 고정됨
- 계산량이 grid_sample보다 많음

---

## 방법 3: Cross-Attention (DETR 스타일)

### 아이디어

각 관절을 query로, backbone spatial features를 key/value로 사용하는 transformer attention.

```python
class JointCrossAttention(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.coord_embed = nn.Linear(2, d_model)
        self.backbone_proj = nn.Linear(2048, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, coords_2d, backbone_feat):
        """
        Args:
            coords_2d: [B, K, 2]
            backbone_feat: [B, C, H, W]
        """
        B, K, _ = coords_2d.shape

        # Query: joint position embeddings
        joint_queries = self.coord_embed(coords_2d)  # [B, K, D]

        # Key/Value: backbone spatial tokens
        backbone_tokens = backbone_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        backbone_kv = self.backbone_proj(backbone_tokens)  # [B, H*W, D]

        # Cross attention
        depth_features, attn_weights = self.cross_attn(
            query=joint_queries,
            key=backbone_kv,
            value=backbone_kv
        )  # [B, K, D], [B, K, H*W]

        return depth_features, attn_weights
```

### 장단점

**장점**:
- 학습 가능한 attention으로 최적 위치 탐색
- 여러 위치에서 정보 통합 가능
- Attention map 시각화로 해석 가능

**단점**:
- 구현 복잡
- 파라미터 증가
- 학습이 불안정할 수 있음

---

## 방법 4: Deformable Attention

### 아이디어

DETR의 deformable attention처럼, 각 관절이 학습된 offset으로 여러 위치를 sparse하게 샘플링.

```python
# 각 관절마다 K개의 sampling point
# offset을 학습하여 최적 위치 탐색
sampling_offsets = self.offset_network(coords_2d)  # [B, K, num_points, 2]
sampling_locations = coords_2d.unsqueeze(2) + sampling_offsets

# 각 위치에서 feature 샘플링 후 weighted sum
```

### 장단점

**장점**: Sparse sampling으로 효율적, 유연한 receptive field

**단점**: 구현 복잡, 학습 어려움

---

## 구현 우선순위

| 순위 | 방법 | 이유 |
|------|------|------|
| 1 | **Grid Sampling** | 간단, 직관적, 빠른 검증 |
| 2 | Heatmap Weighted | Grid의 soft 버전 |
| 3 | Cross-Attention | 성능 개선 여지 있을 때 |

---

## 다음 단계

1. [ ] Grid Sampling 기반 `CustomEgoposeSpatialLiftingHead` 구현
2. [ ] Smoke test로 훈련 파이프라인 검증
3. [ ] Full dataset 훈련 및 baseline 비교
4. [ ] 필요시 방법 2, 3 시도
