#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn import datasets


class Objective:
    """目的関数に相当するクラス"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        """オブジェクトが呼び出されたときに呼ばれる特殊メソッド"""
        # RandomForest のパラメータを最適化してみる
        params = {
            'n_estimators': 100,
            'max_depth': trial.suggest_int('max_depth', 2, 32),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 16),
        }
        model = RandomForestClassifier(**params)
        # 5-Fold Stratified CV
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(model,
                                X=self.X, y=self.y,
                                cv=kf,
                                # メトリックは符号を反転したロジスティック損失
                                scoring='neg_log_loss',
                                n_jobs=-1)
        return scores['test_score'].mean()


def main():
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target
    objective = Objective(X, y)
    # 関数を最大化するように最適化する
    study = optuna.create_study(direction='maximize')
    # 試行回数ではなく特定の時間内で最適化する
    study.optimize(objective, timeout=60)  # この例では 60 秒
    print('params:', study.best_params)


if __name__ == '__main__':
    main()