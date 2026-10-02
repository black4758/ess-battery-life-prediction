"""
[데이터 전처리 모듈: src/preprocess.py]

역할:
1. 대용량 MATLAB v7.3(.mat) 파일에서 원시 배터리 데이터 로드
2. 수명 결측치(NaN, 음수 등) 및 데이터 품질 불량 셀 필터링
3. 센서 오류 이상치(충전시간 3,933분 등 비정상 스파이크) 정제
4. 정제된 배터리별 시계열 및 전압 곡선 데이터를 표준 딕셔너리 형태로 반환
"""

import os
import h5py
import numpy as np
from typing import List, Dict, Any


def clean_cell_data(cell_idx: int, b_hdf5: Any, f_handle: h5py.File, batch_name: str) -> Dict[str, Any]:
    """
    개별 배터리 셀의 원시 데이터를 검증하고 전처리를 수행합니다.
    데이터에 결측치나 치명적 오류가 있는 경우 None을 반환합니다.
    """
    c_life_ds = b_hdf5['cycle_life']
    cycles_ds = b_hdf5['cycles']
    summary_ds = b_hdf5['summary']
    has_readable = 'policy_readable' in b_hdf5
    pol_ds = b_hdf5['policy_readable'] if has_readable else b_hdf5['policy']

    # 1. 수명(Cycle Life) 결측치 및 비정상값 전처리
    ref_life = c_life_ds[0, cell_idx] if c_life_ds.shape[0] == 1 else c_life_ds[cell_idx, 0]
    val_life = f_handle[ref_life][()] if isinstance(ref_life, h5py.Reference) else ref_life
    cycle_life = float(np.squeeze(val_life))

    # 수명이 기록되지 않았거나(NaN), 0 이하인 불량 셀 필터링 (원논문 기준 정제)
    if np.isnan(cycle_life) or cycle_life <= 0:
        return None

    # 2. 충전 정책(Policy) 문자열 디코딩 및 정제
    ref_pol = pol_ds[0, cell_idx] if pol_ds.shape[0] == 1 else pol_ds[cell_idx, 0]
    if isinstance(ref_pol, h5py.Reference):
        pol_obj = f_handle[ref_pol][()]
        if pol_obj.dtype.kind in ['u', 'i']:
            policy_str = ''.join(chr(c) for c in pol_obj.flatten())
        else:
            policy_str = str(pol_obj)
    else:
        policy_str = str(ref_pol)

    # 3. 사이클 시계열(Qdlin) 검증: 최소 100사이클 이상 데이터가 존재하는지 확인
    ref_cycles = cycles_ds[0, cell_idx] if cycles_ds.shape[0] == 1 else cycles_ds[cell_idx, 0]
    cycles_grp = f_handle[ref_cycles]
    if 'Qdlin' not in cycles_grp:
        return None

    qdlin_ds = cycles_grp['Qdlin']
    n_cycles_avail = qdlin_ds.shape[0] if qdlin_ds.shape[0] > qdlin_ds.shape[1] else qdlin_ds.shape[1]
    
    # 조기 진단 모델은 100사이클 데이터가 필수이므로, 100사이클 미만 셀은 제외
    if n_cycles_avail < 100:
        return None

    # 10번째 및 100번째 사이클의 균일 전압 보간 곡선(Qdlin) 추출 (2.0V ~ 3.6V, 1000개 포인트)
    ref_c10 = qdlin_ds[9, 0] if qdlin_ds.shape[0] > qdlin_ds.shape[1] else qdlin_ds[0, 9]
    ref_c100 = qdlin_ds[99, 0] if qdlin_ds.shape[0] > qdlin_ds.shape[1] else qdlin_ds[0, 99]
    q10 = np.squeeze(f_handle[ref_c10][()] if isinstance(ref_c10, h5py.Reference) else ref_c10)
    q100 = np.squeeze(f_handle[ref_c100][()] if isinstance(ref_c100, h5py.Reference) else ref_c100)

    # 4. Summary 시계열 데이터 추출 및 센서 스파이크 이상치 정제
    ref_sum = summary_ds[0, cell_idx] if summary_ds.shape[0] == 1 else summary_ds[cell_idx, 0]
    sum_grp = f_handle[ref_sum]
    tavg_raw = np.squeeze(sum_grp['Tavg'][()])
    tmax_raw = np.squeeze(sum_grp['Tmax'][()])
    ctime_raw = np.squeeze(sum_grp['chargetime'][()])
    qd_raw = np.squeeze(sum_grp['QDischarge'][()])
    ir_raw = np.squeeze(sum_grp['IR'][()])

    # [이상치 정제 1] 충전시간(chargetime) 센서 글리치 필터링
    # 시험 챔버 전원 중단이나 일시 정지로 충전시간이 3,933분 등으로 튄 비정상 측정치를 30분 이내로 필터링
    valid_ctime = [t for t in ctime_raw[:100] if 0 < t <= 30]
    clean_mean_ctime = float(np.mean(valid_ctime)) if len(valid_ctime) >= 10 else float(np.median(ctime_raw[:100]))

    # [이상치 정제 2] 온도(Tavg) 센서 0도 미만 측정 오류 필터링
    valid_tavg = [t for t in tavg_raw[:100] if t > 0]
    clean_mean_tavg = float(np.mean(valid_tavg)) if len(valid_tavg) >= 10 else np.nan

    # [이상치 정제 3] 방전용량(QDischarge) 초기 100사이클 유효 슬라이싱
    clean_qd = qd_raw[:100]

    return {
        'cell_id': f'{batch_name}_{cell_idx}',
        'batch': batch_name,
        'charging_policy': policy_str,
        'cycle_life': cycle_life,
        'q10': q10,
        'q100': q100,
        'clean_mean_chargetime': clean_mean_ctime,
        'clean_mean_tavg': clean_mean_tavg,
        'clean_qd': clean_qd,
        'ir_raw': ir_raw
    }


def load_and_preprocess_batch(mat_path: str, batch_name: str) -> List[Dict[str, Any]]:
    """
    특정 배치 파일(.mat)을 읽어 유효한 정상 셀들의 전처리된 리스트를 반환합니다.
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"데이터 파일 경로를 찾을 수 없습니다: {mat_path}")

    print(f"[*] [{batch_name}] 데이터 전처리 시작: {mat_path}")
    cleaned_cells = []

    with h5py.File(mat_path, 'r') as f:
        b = f['batch']
        c_life_ds = b['cycle_life']
        n_cells = c_life_ds.shape[0] if c_life_ds.shape[0] > c_life_ds.shape[1] else c_life_ds.shape[1]

        for i in range(n_cells):
            cell_data = clean_cell_data(i, b, f, batch_name)
            if cell_data is not None:
                cleaned_cells.append(cell_data)

    print(f"    -> 총 {n_cells}개 중 결측치/이상치 정제 후 유효 셀: {len(cleaned_cells)}개")
    return cleaned_cells


if __name__ == '__main__':
    # 전처리 모듈 단독 테스트 실행
    sample_b1 = load_and_preprocess_batch('data/2017-05-12_batchdata_updated_struct_errorcorrect.mat', 'Batch 1')
    sample_b2 = load_and_preprocess_batch('data/2018-02-20_batchdata_updated_struct_errorcorrect.mat', 'Batch 2')
    print(f"\n[전처리 완료] Batch 1 유효 셀: {len(sample_b1)}개, Batch 2 유효 셀: {len(sample_b2)}개")
