"""
[피처 엔지니어링 모듈: src/features.py]

역할:
1. src/preprocess.py에서 정제된 깨끗한 배터리 데이터를 입력받음
2. 전기화학 도메인 지식(Nature Energy 2019 논문 기반)에 근거한 핵심 피처 계산:
   - log10_Var_dQ : ΔQ(V) = Q100(V) - Q10(V) 곡선 뒤틀림의 분산 로그값 (수명 예측 1순위 신호)
   - min_dQ       : ΔQ(V) 곡선의 최솟값 (방전 전압 저하 계곡 깊이)
   - mean_chargetime: 초기 100사이클 평균 충전 시간 (충전 부하 신호)
   - mean_Tavg    : 초기 100사이클 평균 온도 (열 스트레스 신호)
   - slope_QD     : 초기 100사이클 방전 용량 감소 기울기 (용량 열화 추세)
   - delta_QD_100_10 : 100사이클과 10사이클 간의 방전 용량 단순 차이
3. 머신러닝 학습용 정형 테이블(DataFrame) 생성 및 data/features.csv로 저장
"""

import os
import sys
import numpy as np
import pandas as pd

# src 폴더 내부 모듈 임포트를 위한 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_preprocess_batch


def calculate_features_for_cells(cleaned_cells: list) -> pd.DataFrame:
    """
    정제된 개별 셀 리스트에서 머신러닝 입력용 핵심 피처들을 계산하여 DataFrame으로 반환합니다.
    """
    records = []

    for cell in cleaned_cells:
        q10 = cell['q10']
        q100 = cell['q100']

        # -------------------------------------------------------------
        # 1. ΔQ(V) 곡선 기반 전기화학 피처 추출
        # 100번째 사이클 곡선에서 10번째 사이클 곡선을 뺀 차분 곡선
        # -------------------------------------------------------------
        dq = q100 - q10
        var_dq = float(np.var(dq))
        # log10(Var(ΔQ)): 분산이 0에 가까울 수 있으므로 epsilon(1e-12) 추가
        log_var_dq = float(np.log10(var_dq + 1e-12))
        min_dq = float(np.min(dq))
        mean_dq = float(np.mean(dq))

        # -------------------------------------------------------------
        # 2. 방전 용량(QDischarge) 시계열 기반 열화 속도 피처
        # -------------------------------------------------------------
        clean_qd = cell['clean_qd']
        cycles_idx = np.arange(1, len(clean_qd) + 1)
        
        # 1차 선형 회귀로 초기 용량 감소 기울기 산출
        if len(clean_qd) >= 10:
            slope_qd = float(np.polyfit(cycles_idx, clean_qd, 1)[0])
        else:
            slope_qd = 0.0

        # 100사이클 - 10사이클 방전용량 차이
        if len(clean_qd) >= 100:
            delta_qd = float(clean_qd[99] - clean_qd[9])
        else:
            delta_qd = 0.0

        records.append({
            'cell_id': cell['cell_id'],
            'batch': cell['batch'],
            'charging_policy': cell['charging_policy'],
            'cycle_life': cell['cycle_life'],
            # 핵심 피처 1: 전압 곡선 분산 (수명과 가장 강력한 상관성, r = -0.85)
            'log10_Var_dQ': log_var_dq,
            # 핵심 피처 2: 전압 곡선 최솟값 (2.4~3.0V 저전압 계곡 깊이)
            'min_dQ': min_dq,
            # 보조 피처: 전압 곡선 평균값
            'mean_dQ': mean_dq,
            # 핵심 피처 3: 평균 충전 시간 (센서 노이즈 필터링 완료된 유효 충전 시간)
            'mean_chargetime': cell['clean_mean_chargetime'],
            # 핵심 피처 4: 평균 온도 (초기 100사이클 누적 열 부하)
            'mean_Tavg': cell['clean_mean_tavg'],
            # 보조 피처: 초기 용량 감소 추세 기울기
            'slope_QD': slope_qd,
            # 보조 피처: 용량 단순 차이
            'delta_QD_100_10': delta_qd
        })

    return pd.DataFrame(records)


def run_feature_pipeline(data_dir: str = 'data', output_csv: str = 'data/features.csv') -> pd.DataFrame:
    """
    전체 전처리 및 피처 엔지니어링 파이프라인을 실행하고 결과를 CSV로 저장합니다.
    """
    b1_path = os.path.join(data_dir, '2017-05-12_batchdata_updated_struct_errorcorrect.mat')
    b2_path = os.path.join(data_dir, '2018-02-20_batchdata_updated_struct_errorcorrect.mat')

    # 1. preprocess.py를 호출하여 원시 데이터 정제 수행
    print("==================================================================")
    print("🚀 [Step 1] 원시 데이터 전처리 (preprocess.py)")
    print("==================================================================")
    cleaned_b1 = load_and_preprocess_batch(b1_path, 'Batch 1')
    cleaned_b2 = load_and_preprocess_batch(b2_path, 'Batch 2')

    # 2. 정제된 데이터에서 핵심 피처 계산
    print("\n==================================================================")
    print("🔬 [Step 2] 핵심 전기화학 피처 추출 (features.py)")
    print("==================================================================")
    df_b1_feats = calculate_features_for_cells(cleaned_b1)
    df_b2_feats = calculate_features_for_cells(cleaned_b2)
    print(f"[*] Batch 1 피처 계산 완료: {len(df_b1_feats)}행 x {df_b1_feats.shape[1]}열")
    print(f"[*] Batch 2 피처 계산 완료: {len(df_b2_feats)}행 x {df_b2_feats.shape[1]}열")

    # 3. 통합 및 CSV 저장
    df_all_feats = pd.concat([df_b1_feats, df_b2_feats], ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_all_feats.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"\n[+] 머신러닝용 최종 피처 테이블 저장 완료: {output_csv} (총 {len(df_all_feats)}개 셀)")
    return df_all_feats


if __name__ == '__main__':
    run_feature_pipeline()
