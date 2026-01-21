# 코드 수정 필요 사항

> 최종 업데이트: 2026-01-21

---

## 1. Metric squeeze() 버그 (Critical)

**파일**: `mmpose/evaluation/metrics/custom_xr_egopose_metric.py`
**위치**: line 214-215

### 현재 코드
```python
pred_batch_3d_keypoints = torch.stack(pred_batch_3d_keypoints).squeeze()
gt_batch_keypoint_3d = torch.stack(gt_batch_keypoint_3d).squeeze()
```

### 문제
`squeeze()`가 모든 크기 1인 차원을 제거하여 N=1일 때 batch 차원이 사라짐.

| 샘플 수 | stack 후 | squeeze 후 | 예상 | 결과 |
|---------|----------|------------|------|------|
| N > 1 | [N, 1, 16, 3] | [N, 16, 3] | OK | ✅ |
| **N = 1** | [1, 1, 16, 3] | **[16, 3]** | [1, 16, 3] | ❌ |

### 영향
- N=1일 때 16개 관절이 16개 샘플로 잘못 해석됨
- MPJPE 계산 완전히 틀림

### 수정안
```python
# squeeze(dim=1)로 instance 차원만 제거
pred_batch_3d_keypoints = torch.stack(pred_batch_3d_keypoints).squeeze(dim=1)
gt_batch_keypoint_3d = torch.stack(gt_batch_keypoint_3d).squeeze(dim=1)
```

### 우선순위
**Critical** - 단일 샘플 테스트 시 반드시 발생

---

## 2. Transpose 로직 취약

**파일**: `mmpose/evaluation/metrics/mo2cap2_evaluate.py`
**위치**: `compute_error()` 함수, line 616-621

### 현재 코드
```python
if pred.shape[1] != 3:
    pred = np.transpose(pred, [1, 0])

if gt.shape[1] != 3:
    gt = np.transpose(gt, [1, 0])
```

### 문제
- 2D 배열 `[16, 3]` 또는 `[3, 16]`만 가정
- 3D 배열 입력 시 예상치 못한 동작
- shape 검증 없이 암묵적 변환

### 수정안
```python
# 명시적 shape 검증 추가
assert pred.ndim == 2, f"Expected 2D array, got {pred.ndim}D with shape {pred.shape}"
assert pred.shape == (16, 3) or pred.shape == (3, 16), f"Unexpected shape: {pred.shape}"

if pred.shape[1] != 3:
    pred = np.transpose(pred, [1, 0])
```

### 우선순위
**Medium** - 비표준 입력 시에만 발생

---

## 3. Shape 검증 부재

**파일**: `mmpose/evaluation/metrics/custom_xr_egopose_metric.py`
**위치**: `process()` 및 `compute_metrics()`

### 문제
- 입력 keypoint3d shape에 대한 검증 없음
- 잘못된 shape 입력 시 silent error

### 수정안
```python
# process()에서 검증 추가
pred['keypoint3d'] = data_sample['pred_instances']['keypoint_3d']
assert pred['keypoint3d'].shape[-2:] == (16, 3), \
    f"Expected keypoint3d shape [..., 16, 3], got {pred['keypoint3d'].shape}"
```

### 우선순위
**Low** - 디버깅 용이성 개선

---

## 수정 체크리스트

| # | 파일 | 문제 | 우선순위 | 상태 |
|---|------|------|----------|------|
| 1 | `custom_xr_egopose_metric.py:214-215` | squeeze() 버그 | Critical | ✅ 수정완료 |
| 2 | `mo2cap2_evaluate.py:616-621` | transpose 취약 | Medium | ✅ 수정완료 |
| 3 | `custom_xr_egopose_metric.py` | shape 검증 부재 | Low | ✅ 수정완료 |

> 수정일: 2026-01-21

---

## 테스트 방법

### squeeze 버그 확인
```python
# 단일 샘플로 validation 실행
val_dataloader = dict(batch_size=1, ...)
# MPJPE가 비정상적으로 크면 버그 발생
```

### 수정 후 검증
```bash
# 단일 샘플 테스트
python tools/test.py <config> <checkpoint> --cfg-options val_dataloader.batch_size=1
```
