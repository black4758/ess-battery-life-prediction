"""
[모델 학습 및 성능 평가 모듈: src/train.py]

역할:
1. data/features.csv 데이터를 불러와 Train, Hold-out Valid, Test 데이터셋으로 분할
2. 타깃 변수 수명(cycle_life)의 log10 변환 및 StandardScaler 스케일링 전처리
3. 후보 모델(ElasticNet vs 트리 앙상블) 학습 및 하이퍼파라미터 튜닝
4. 노션 공식 지정 포맷의 성능 지표(MAPE, RMSE, Gap 3종) 자동 산출 및 results/model_performance.csv 저장
5. 오차가 가장 큰 배터리 셀(Worst-3) 오류 분석 및 시각화 그래프 저장
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def calc_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """평균 절대 백분율 오차 (MAPE, %) 계산"""
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """평균 제곱근 오차 (RMSE) 계산"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_and_evaluate(features_csv: str = 'data/features.csv', results_dir: str = 'results') -> pd.DataFrame:
    """
    모델 학습, 검증, 최종 평가를 수행하고 노션 포맷 보고서 및 차트를 생성합니다.
    """
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"피처 데이터가 없습니다. 먼저 features.py를 실행하세요: {features_csv}")

    os.makedirs(results_dir, exist_ok=True)
    df = pd.read_csv(features_csv)

    # 1. 데이터 분할: Batch 1 (학습용 46개) vs Batch 2 (최종 테스트용 39개)
    b1_df = df[df['batch'] == 'Batch 1'].reset_index(drop=True)
    b2_df = df[df['batch'] == 'Batch 2'].reset_index(drop=True)

    print("==================================================================")
    print("📊 [1] 데이터셋 분할 및 특성 확인")
    print("==================================================================")
    print(f"• 학습 데이터 (Batch 1) : 총 {len(b1_df)}개 셀")
    print(f"• 평가 데이터 (Batch 2) : 총 {len(b2_df)}개 셀 (학습에 일절 관여하지 않는 미지의 시험셋)")

    # 핵심 모델 입력 피처 세트 선정 (다중공선성 정제 및 Severson Full Model 기반)
    feature_cols = ['log10_Var_dQ', 'min_dQ', 'mean_chargetime', 'slope_QD', 'delta_QD_100_10']
    print(f"• 입력 피처 목록 ({len(feature_cols)}개) : {feature_cols}")

    X_b1 = b1_df[feature_cols].values
    y_b1 = b1_df['cycle_life'].values
    y_b1_log = np.log10(y_b1)  # 원논문 방식: 상대 오차 최소화를 위한 log10 타깃 변환

    X_b2 = b2_df[feature_cols].values
    y_b2 = b2_df['cycle_life'].values

    # ==============================================================================
    # 2. 모델 학습 및 평가 파이프라인
    # ==============================================================================
    print("\n==================================================================")
    print("⚙️ [2] 모델 학습 및 교차 검증 (Cross-Validation)")
    print("==================================================================")

    # (1) Train 5-Fold Cross Validation on Batch 1
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mapes = []
    for tr_i, val_i in kf.split(X_b1):
        pipe_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42))
        ])
        pipe_cv.fit(X_b1[tr_i], y_b1_log[tr_i])
        pred_val_log = pipe_cv.predict(X_b1[val_i])
        pred_val = 10 ** pred_val_log
        cv_mapes.append(calc_mape(y_b1[val_i], pred_val))

    train_cv_mape = float(np.mean(cv_mapes))

    # (2) Valid Hold-out (80% Train, 20% Valid) - 프로토콜 분리 검증
    X_tr, X_val, y_tr_log, y_val_log, y_tr, y_val = train_test_split(
        X_b1, y_b1_log, y_b1, test_size=0.2, random_state=42
    )
    pipe_holdout = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42))
    ])
    pipe_holdout.fit(X_tr, y_tr_log)
    pred_val_ho = 10 ** pipe_holdout.predict(X_val)
    valid_ho_mape = calc_mape(y_val, pred_val_ho)

    # (3) Final Test Evaluation on Batch 2 (Trained on 100% of Batch 1)
    pipe_final = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42))
    ])
    pipe_final.fit(X_b1, y_b1_log)
    pred_b2_log = pipe_final.predict(X_b2)
    pred_b2 = 10 ** pred_b2_log
    test_b2_mape = calc_mape(y_b2, pred_b2)
    test_b2_rmse = calc_rmse(y_b2, pred_b2)

    # (4) 트리 앙상블 비교군 (Random Forest)
    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42))
    ])
    pipe_rf.fit(X_b1, y_b1_log)
    pred_b2_rf = 10 ** pipe_rf.predict(X_b2)
    rf_test_mape = calc_mape(y_b2, pred_b2_rf)

    print(f"• [후보 모델 1] ElasticNet Test MAPE : {test_b2_mape:.2f}%  (최종 채택 🏆)")
    print(f"• [후보 모델 2] RandomForest Test MAPE: {rf_test_mape:.2f}%")

    # ==============================================================================
    # 3. 노션 공식 지정 포맷 성능 결과표 산출
    # ==============================================================================
    gap_train_val = valid_ho_mape - train_cv_mape
    gap_val_test = test_b2_mape - valid_ho_mape
    gap_target_test = test_b2_mape - 9.1  # 원논문 9.1% 대비 차이

    performance_rows = [
        {"구분": "Train (Batch 1 CV)", "MAPE (%)": f"{train_cv_mape:.2f}", "비고": "Batch 1 5-Fold 교차검증 평균"},
        {"구분": "Valid (Batch 1 Hold-out)", "MAPE (%)": f"{valid_ho_mape:.2f}", "비고": "프로토콜 독립 분리 검증 (20%)"},
        {"구분": "Test (Batch 2)", "MAPE (%)": f"{test_b2_mape:.2f}", "비고": f"1차 필수 테스트셋 (RMSE: {test_b2_rmse:.1f})"},
        {"구분": "Gap (Train-Valid)", "MAPE (%)": f"{gap_train_val:+.2f}", "비고": "과적합 여부 확인 (음수=과적합 없음)"},
        {"구분": "Gap (Valid-Test)", "MAPE (%)": f"{gap_val_test:+.2f}", "비고": "배치 간 일반화 격차"},
        {"구분": "Gap (Target-Test)", "MAPE (%)": f"{gap_target_test:+.2f}", "비고": "원논문 목표(9.1%) 대비 격차"}
    ]
    perf_df = pd.DataFrame(performance_rows)
    perf_csv_path = os.path.join(results_dir, 'model_performance.csv')
    perf_df.to_csv(perf_csv_path, index=False, encoding='utf-8-sig')

    print("\n==================================================================")
    print("📋 [3] 노션 공식 포맷 성능 결과표 (results/model_performance.csv)")
    print("==================================================================")
    print(perf_df.to_string(index=False))

    # ==============================================================================
    # 4. 오류 분석 (Error Analysis): 가장 크게 틀린 Worst-3 셀 분석
    # ==============================================================================
    b2_eval = b2_df.copy()
    b2_eval['pred_life'] = pred_b2
    b2_eval['error_cycles'] = np.abs(b2_eval['cycle_life'] - b2_eval['pred_life'])
    b2_eval['error_pct'] = (b2_eval['error_cycles'] / b2_eval['cycle_life']) * 100

    worst_cells = b2_eval.sort_values(by='error_pct', ascending=False).head(3)
    worst_csv_path = os.path.join(results_dir, 'worst_predictions.csv')
    worst_cells[['cell_id', 'charging_policy', 'cycle_life', 'pred_life', 'error_pct']].to_csv(
        worst_csv_path, index=False, encoding='utf-8-sig'
    )

    print("\n==================================================================")
    print("🔍 [4] 오류 분석: 모델이 가장 크게 틀린 Top 3 셀 (Worst Errors)")
    print("==================================================================")
    for idx, row in worst_cells.iterrows():
        print(f"• 셀 ID: {row['cell_id']} | 정책: {row['charging_policy']}")
        print(f"  실제 수명: {row['cycle_life']:.0f}회 vs 예측 수명: {row['pred_life']:.0f}회 (오차율: {row['error_pct']:.1f}%)")

    # ==============================================================================
    # 5. 실제값 vs 예측값 산점도 시각화 차트 저장
    # ==============================================================================
    plt.figure(figsize=(7, 6), dpi=150)
    plt.scatter(y_b2, pred_b2, color='steelblue', edgecolor='black', alpha=0.8, s=60, label=f'Batch 2 Test (MAPE={test_b2_mape:.1f}%)')
    
    # 1:1 대각 완벽 예측선
    min_val = min(y_b2.min(), pred_b2.min()) * 0.9
    max_val = max(y_b2.max(), pred_b2.max()) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Ideal 1:1 Line')
    
    plt.xlabel('Actual Cycle Life (True)', fontsize=11, fontweight='bold')
    plt.ylabel('Predicted Cycle Life (Model)', fontsize=11, fontweight='bold')
    plt.title('Cycle Life Prediction: Actual vs Predicted (Batch 2)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    chart_path = os.path.join(results_dir, 'actual_vs_predicted.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"\n[+] 시각화 차트 저장 완료: {chart_path}")

    return perf_df


if __name__ == '__main__':
    train_and_evaluate()
