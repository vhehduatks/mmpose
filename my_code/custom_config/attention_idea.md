

## 제안: Attention 기반 Lifting Network

### 현재 구조의 한계

```python
# 현재: 단순 concat → FC
x = concat([coords_2d, confidence, z_backbone, hmd])  # [B, 313]
pose_3d = FC(x)  # [B, 48]
```

**문제점**:
- 모든 입력을 단순 concat → 관계 모델링 부재
- 어떤 관절이 어떤 depth 정보를 참조해야 하는지 불명확
- 관절 간 구조적 관계 (대칭, 부모-자식) 무시

### 방안 1: Joint Self-Attention

```
2D coords [B, 16, 2] → Linear → Joint tokens [B, 16, D]
                                      ↓
                              Self-Attention
                              (관절 간 관계 학습)
                                      ↓
                              [B, 16, D]
                                      ↓
                                  3D Pose
```

**효과**:
- 왼팔 ↔ 오른팔 대칭 관계 학습
- 부모-자식 관절 관계 (어깨→팔꿈치→손목)
- 가려진 관절이 보이는 관절 참조

---

### 방안 2: Cross-Attention (2D → Backbone) ⭐ 추천

**핵심 아이디어**: 각 관절의 2D 위치가 Backbone feature에서 해당 depth 정보를 쿼리

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

**구현 예시**:
```python
class CrossAttentionLifting(nn.Module):
    def __init__(self, joint_dim=64, backbone_dim=256, num_heads=4):
        super().__init__()

        # 2D coords → joint queries [B, 16, 64]
        self.joint_embed = nn.Sequential(
            nn.Linear(2, joint_dim),
            nn.ReLU()
        )

        # Backbone spatial tokens [B, 64, D]
        self.backbone_proj = nn.Conv2d(2048, backbone_dim, 1)
        self.kv_proj = nn.Linear(backbone_dim, joint_dim)

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Output
        self.output_proj = nn.Linear(joint_dim, 3)

    def forward(self, coords_2d, backbone_feat, confidence, hmd_info):
        B = coords_2d.size(0)

        # Joint queries from 2D coords
        joint_q = self.joint_embed(coords_2d)  # [B, 16, 64]

        # Backbone spatial tokens
        backbone_tokens = self.backbone_proj(backbone_feat)  # [B, 256, 8, 8]
        backbone_tokens = backbone_tokens.flatten(2).transpose(1, 2)  # [B, 64, 256]
        backbone_kv = self.kv_proj(backbone_tokens)  # [B, 64, 64]

        # Cross attention: 각 관절이 backbone에서 depth 정보 쿼리
        depth_features, attn_weights = self.cross_attn(
            query=joint_q,      # [B, 16, 64] - 각 관절
            key=backbone_kv,    # [B, 64, 64] - spatial positions
            value=backbone_kv
        )
        # depth_features: [B, 16, 64] - depth-aware joint features
        # attn_weights: [B, 16, 64] - 어느 위치 참조했는지 시각화 가능!

        # Combine with confidence and HMD
        # ... (추가 처리)

        pose_3d = self.output_proj(depth_features)  # [B, 16, 3]
        return pose_3d, attn_weights
```

**장점**:
1. **선택적 depth 쿼리**: 각 관절이 필요한 spatial 위치에서 depth 정보 가져옴
2. **해석 가능**: `attn_weights`로 "어느 위치에서 depth를 가져왔는지" 시각화
3. **역할 분리와 일치**: 2D=Query(위치), Backbone=Key/Value(depth)

---

### 방안 3: Joint-HMD Cross-Attention

```
Query: Joint tokens [B, 16, D]
Key/Value: HMD tokens [B, 3, D]  (head, right_hand, left_hand)
                ↓
        Cross-Attention
                ↓
    "손 관절 → 손 HMD 참조"
    "몸통 관절 → head HMD 참조"
```

**구현**:
```python
class JointHMDCrossAttention(nn.Module):
    def __init__(self, joint_dim=64, num_heads=4):
        super().__init__()

        # HMD를 3개 token으로 (head, right_hand, left_hand)
        self.hmd_embed = nn.Sequential(
            nn.Linear(9, joint_dim * 3),  # → [B, 3, D]
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, joint_features, hmd_info):
        B = joint_features.size(0)

        # HMD → 3 tokens [B, 3, D]
        hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)

        # Cross attention: joints attend to HMD
        enhanced, _ = self.cross_attn(
            query=joint_features,  # [B, 16, D]
            key=hmd_tokens,        # [B, 3, D]
            value=hmd_tokens
        )

        return joint_features + enhanced  # Residual
```

**효과**: 손 관절 → 손 HMD, 몸통 → head HMD로 자연스러운 매핑

---

### 전체 Attention Lifting 구조 (추천)

```
                    Backbone feat [2048, 8, 8]
                           │
         ┌─────────────────┴─────────────────┐
         ↓                                   ↓
    Heatmap [16,47,47]              Backbone tokens [64, D]
         ↓                                   │
    soft_argmax                              │
         ↓                                   │
   2D coords [16, 2]                         │
         ↓                                   │
   Joint embed [16, D]                       │
         │                                   │
         └───────── Cross-Attention ─────────┘
                   (Q: joints, K/V: backbone)
                           ↓
                  Depth-aware joints [16, D]
                           │
                           + HMD Cross-Attention
                           ↓
                    Self-Attention
                   (관절 간 관계)
                           ↓
                    Output proj
                           ↓
                    3D Pose [16, 3]
```

### Gradient 흐름 설계

```python
def forward(self, heatmaps, backbone_feat, hmd_info):
    # 2D coords (gradient 차단 - 2D 역할만)
    coords_2d, confidence = soft_argmax_2d(heatmaps)
    coords_2d = coords_2d.detach()
    confidence = confidence.detach()

    # Joint queries
    joint_q = self.joint_embed(coords_2d)  # [B, 16, D]

    # Backbone key/value (gradient 흐름 - 3D 학습)
    backbone_kv = self.backbone_to_tokens(backbone_feat)  # [B, 64, D]

    # Cross attention: 2D 위치 기반 depth 쿼리
    depth_joints, attn = self.cross_attn(joint_q, backbone_kv, backbone_kv)

    # HMD fusion
    hmd_enhanced = self.hmd_cross_attn(depth_joints, hmd_info)

    # Self attention (관절 간 관계)
    refined = self.self_attn(hmd_enhanced)

    # Output
    pose_3d = self.output_proj(refined)

    return pose_3d, attn  # attn으로 해석 가능
```

### 방안 비교

| 방식 | 관절 관계 | Depth 쿼리 | 해석 가능 | 파라미터 |
|------|----------|-----------|----------|---------|
| FC (현재) | ✗ | ✗ | ✗ | ~4M |
| Self-Attention | ✓ | ✗ | △ | ~5M |
| **Cross-Attention** | △ | **✓** | **✓** | ~6M |
| Full (Self+Cross+HMD) | ✓ | ✓ | ✓ | ~8M |

### 구현 우선순위

1. **Cross-Attention (2D → Backbone)** - 핵심, 먼저 구현
2. HMD Cross-Attention - 추가 개선
3. Self-Attention - 필요시

**예상 파일**: `custom_egopose_attention_lifting_head.py`

---
