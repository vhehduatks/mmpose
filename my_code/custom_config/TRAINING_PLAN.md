# 훈련 실행 계획

## 현재 상황 (2026-01-20)

| Model | Full Body MPJPE | Config |
|-------|-----------------|--------|
| **Single COCO** | **42.07mm** | `HMD_xregopose_single_coco_full_config.py` |
| Dual COCO+MPII | 44.91mm | `HMD_xregopose_h5cache_coco_mpii_config.py` |

**목표**: Dual mutual learning으로 42mm 이하 달성

---

## 실험 1: Progressive Warmup (구현 완료 ✅)

**가설**: 초기 epoch에서 pretrained knowledge 보존 → 성능 개선

**Config**: `HMD_xregopose_h5cache_coco_mpii_warmup_config.py`

**실행 명령**:
```bash
cd /home/hyeonghwan/github/mmpose
python tools/train.py my_code/custom_config/HMD_xregopose_h5cache_coco_mpii_warmup_config.py
```

**설정**:
- Head: `CustomxRegoposeBaselinel1_multi_backbone_v2`
- mutual_warmup_epochs: 5 (mutual learning 없음)
- mutual_rampup_epochs: 10 (0→1 선형 증가)
- max_epochs: 20

**예상 로그**:
```
[MutualLearningWarmupHook] Epoch 0: mutual_learning_weight = 0.000
[MutualLearningWarmupHook] Epoch 5: mutual_learning_weight = 0.000
[MutualLearningWarmupHook] Epoch 10: mutual_learning_weight = 0.500
[MutualLearningWarmupHook] Epoch 15: mutual_learning_weight = 1.000
```

**Wandb 프로젝트**: `mmpose_xregopose_coco_mpii_warmup`

---

## 실험 2: Ensemble Teacher (미구현)

**가설**: Confidence 기반 앙상블 teacher가 더 나은 knowledge transfer

**구현 필요**:
- `custom_egopose_baselinel1_head_multi_backbone_v3.py`
- Config: `HMD_xregopose_h5cache_coco_mpii_ensemble_config.py`

---

## 실험 3: Heatmap KL Divergence (미구현)

**가설**: MSE 대신 KL divergence가 soft target으로 더 효과적

**구현 필요**:
- `custom_egopose_baselinel1_head_multi_backbone_v4.py`
- Config: `HMD_xregopose_h5cache_coco_mpii_kldiv_config.py`

---

## 실행 순서

| 순서 | 실험 | 상태 | 명령어 |
|------|------|------|--------|
| 1 | Progressive Warmup | ✅ 준비됨 | `python tools/train.py my_code/custom_config/HMD_xregopose_h5cache_coco_mpii_warmup_config.py` |
| 2 | Ensemble Teacher | ❌ 미구현 | - |
| 3 | KL Divergence | ❌ 미구현 | - |

---

## 결과 기록

### 실험 1: Progressive Warmup

| Metric | Value |
|--------|-------|
| Full Body MPJPE | |
| Upper Body MPJPE | |
| Lower Body MPJPE | |
| Best Epoch | |
| Wandb Run | |

**관찰**:
-

**결론**:
-

---

## 파일 목록

### 새로 생성된 파일 (이번 세션)
- `mmpose/models/heads/heatmap_heads/custom_egopose_baselinel1_head_multi_backbone_v2.py`
- `mmpose/engine/hooks/mutual_learning_hook.py`
- `my_code/custom_config/HMD_xregopose_h5cache_coco_mpii_warmup_config.py`

### 수정된 파일
- `mmpose/models/heads/heatmap_heads/__init__.py` (v2 등록)
- `mmpose/engine/hooks/__init__.py` (hook 등록)
