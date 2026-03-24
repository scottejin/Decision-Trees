from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


@dataclass
class CandidateResult:
    feature_space: str
    model_name: str
    weighted_f1: float
    threshold: float | None


def _load_data(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(train_path), pd.read_csv(test_path)


def _core_feature_space(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_processed = train_df.copy()
    test_processed = test_df.copy()

    age_median = train_processed["Age"].median()
    fare_median = train_processed["Fare"].median()
    embarked_mode = train_processed["Embarked"].mode().iloc[0]

    for frame in (train_processed, test_processed):
        frame["Age"] = frame["Age"].fillna(age_median)
        frame["Fare"] = frame["Fare"].fillna(fare_median)
        frame["Embarked"] = frame["Embarked"].fillna(embarked_mode)
        frame.drop(columns=["Cabin", "PassengerId", "Name", "Ticket"], inplace=True, errors="ignore")

    combined = pd.concat(
        [train_processed.drop(columns=["Survived"]), test_processed],
        axis=0,
        ignore_index=True,
    )
    combined = pd.get_dummies(combined, columns=["Sex", "Embarked"], drop_first=True)
    combined = combined.reindex(columns=["Pclass", "Age", "Fare", "Sex_male"], fill_value=0)

    train_features = combined.iloc[: len(train_df)].reset_index(drop=True)
    test_features = combined.iloc[len(train_df):].reset_index(drop=True)
    return train_features, test_features


def _engineered_feature_space(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train_df.drop(columns=["Survived"]), test_df],
        axis=0,
        ignore_index=True,
    )

    combined["Title"] = (
        combined["Name"]
        .str.extract(r",\s*([^.]+)\.", expand=False)
        .fillna("Unknown")
        .str.strip()
    )
    combined["Title"] = combined["Title"].replace(
        {
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
            "Lady": "Rare",
            "Countess": "Rare",
            "Capt": "Rare",
            "Col": "Rare",
            "Don": "Rare",
            "Dr": "Rare",
            "Major": "Rare",
            "Rev": "Rare",
            "Sir": "Rare",
            "Jonkheer": "Rare",
            "Dona": "Rare",
        }
    )
    combined["Title"] = combined["Title"].where(
        combined["Title"].isin(["Mr", "Miss", "Mrs", "Master"]),
        "Rare",
    )
    combined["Embarked"] = combined["Embarked"].fillna(combined["Embarked"].mode().iloc[0])
    combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
    combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)
    combined["CabinKnown"] = combined["Cabin"].notna().astype(int)
    combined["Deck"] = combined["Cabin"].str[0].fillna("U")
    combined["NameLength"] = combined["Name"].str.len()
    combined["TicketPrefix"] = (
        combined["Ticket"].str.replace(r"[.\d/ ]", "", regex=True).replace("", "NONE")
    )
    ticket_counts = combined["Ticket"].value_counts()
    combined["TicketGroupSize"] = combined["Ticket"].map(ticket_counts).fillna(1)
    combined["FarePerPerson"] = combined["Fare"] / combined["TicketGroupSize"].replace(0, 1)
    combined["AgeMissing"] = combined["Age"].isna().astype(int)

    age_groups = combined.groupby(["Sex", "Pclass", "Title"])["Age"].transform("median")
    combined["Age"] = combined["Age"].fillna(age_groups).fillna(combined["Age"].median())
    combined["Fare"] = (
        combined["Fare"]
        .fillna(combined.groupby("Pclass")["Fare"].transform("median"))
        .fillna(combined["Fare"].median())
    )
    combined["Child"] = (combined["Age"] < 16).astype(int)
    combined["Mother"] = (
            (combined["Sex"] == "female")
            & (combined["Parch"] > 0)
            & (combined["Age"] > 18)
            & (combined["Title"] != "Miss")
    ).astype(int)

    selected_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "FamilySize",
        "IsAlone",
        "CabinKnown",
        "Deck",
        "NameLength",
        "TicketPrefix",
        "TicketGroupSize",
        "FarePerPerson",
        "AgeMissing",
        "Child",
        "Mother",
    ]

    engineered = combined[selected_columns]
    engineered = pd.get_dummies(
        engineered,
        columns=["Sex", "Embarked", "Title", "Deck", "TicketPrefix"],
        drop_first=False,
    )
    engineered.columns = [
        column.replace("[", "_")
        .replace("]", "_")
        .replace("<", "lt")
        .replace(",", "_")
        .replace(" ", "_")
        .replace("(", "_")
        .replace(")", "_")
        for column in engineered.columns
    ]

    train_features = engineered.iloc[: len(train_df)].reset_index(drop=True)
    test_features = engineered.iloc[len(train_df):].reset_index(drop=True)
    return train_features, test_features


def _build_feature_spaces(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[
    str, tuple[pd.DataFrame, pd.DataFrame]]:
    return {
        "core_bonus2_features": _core_feature_space(train_df, test_df),
        "engineered_features": _engineered_feature_space(train_df, test_df),
    }


def _build_candidate_models() -> dict[str, object]:
    return {
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            subsample=0.7,
            n_estimators=600,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features=None,
            max_depth=2,
            learning_rate=0.02,
        ),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=600,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            random_state=RANDOM_STATE,
            n_estimators=800,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }


def _score_thresholds(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.linspace(0.35, 0.65, 61):
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions, average="weighted")
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)

    return best_threshold, best_score


def _evaluate_model(
        feature_space_name: str,
        model_name: str,
        model: object,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
) -> CandidateResult:
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_val)[:, 1]
        threshold, score = _score_thresholds(y_val, probabilities)
        return CandidateResult(feature_space_name, model_name, score, threshold)

    predictions = model.predict(X_val)
    score = f1_score(y_val, predictions, average="weighted")
    return CandidateResult(feature_space_name, model_name, float(score), None)


def _evaluate_feature_space(
        feature_space_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
) -> tuple[list[CandidateResult], list[str]]:
    base_models = _build_candidate_models()
    results = [
        _evaluate_model(feature_space_name, model_name, clone(model), X_train, y_train, X_val, y_val)
        for model_name, model in base_models.items()
    ]
    results.sort(key=lambda item: item.weighted_f1, reverse=True)

    top_model_names = [item.model_name for item in results[:3]]
    top_estimators = [(name, clone(base_models[name])) for name in top_model_names]

    hard_vote = VotingClassifier(estimators=top_estimators, voting="hard", n_jobs=1)
    hard_vote.fit(X_train, y_train)
    hard_predictions = hard_vote.predict(X_val)
    results.append(
        CandidateResult(
            feature_space_name,
            "voting_hard_top3",
            float(f1_score(y_val, hard_predictions, average="weighted")),
            None,
        )
    )

    soft_vote = VotingClassifier(
        estimators=top_estimators,
        voting="soft",
        weights=[3, 2, 1],
        n_jobs=1,
    )
    soft_vote.fit(X_train, y_train)
    soft_probabilities = soft_vote.predict_proba(X_val)[:, 1]
    soft_threshold, soft_score = _score_thresholds(y_val, soft_probabilities)
    results.append(
        CandidateResult(
            feature_space_name,
            "voting_soft_top3_weighted",
            soft_score,
            soft_threshold,
        )
    )

    results.sort(key=lambda item: item.weighted_f1, reverse=True)
    return results, top_model_names


def _fit_selected_model(
        selected_result: CandidateResult,
        top_model_names: list[str],
) -> object:
    base_models = _build_candidate_models()

    if selected_result.model_name == "voting_hard_top3":
        estimators = [(name, clone(base_models[name])) for name in top_model_names]
        return VotingClassifier(estimators=estimators, voting="hard", n_jobs=1)

    if selected_result.model_name == "voting_soft_top3_weighted":
        estimators = [(name, clone(base_models[name])) for name in top_model_names]
        return VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=[3, 2, 1],
            n_jobs=1,
        )

    return clone(base_models[selected_result.model_name])


def _results_to_frame(results: list[CandidateResult]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "feature_space": result.feature_space,
                "model": result.model_name,
                "weighted_f1": result.weighted_f1,
                "threshold": result.threshold,
            }
            for result in results
        ]
    )
    return frame.sort_values("weighted_f1", ascending=False).reset_index(drop=True)


def run_bonus2_workflow(
        train_path: str | Path = "train.csv",
        test_path: str | Path = "test.csv",
        submission_path: str | Path = "bonus2_improved_submission.csv",
) -> dict[str, object]:
    train_df, test_df = _load_data(train_path, test_path)
    y = train_df["Survived"].copy()
    feature_spaces = _build_feature_spaces(train_df, test_df)

    train_indices, val_indices = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    all_results: list[CandidateResult] = []
    top_model_names_by_space: dict[str, list[str]] = {}

    for feature_space_name, (train_features, _) in feature_spaces.items():
        feature_results, top_model_names = _evaluate_feature_space(
            feature_space_name,
            train_features.iloc[train_indices].reset_index(drop=True),
            y.iloc[train_indices].reset_index(drop=True),
            train_features.iloc[val_indices].reset_index(drop=True),
            y.iloc[val_indices].reset_index(drop=True),
        )
        all_results.extend(feature_results)
        top_model_names_by_space[feature_space_name] = top_model_names

    leaderboard = _results_to_frame(all_results)
    selected_row = leaderboard.iloc[0]
    selected_result = CandidateResult(
        feature_space=selected_row["feature_space"],
        model_name=selected_row["model"],
        weighted_f1=float(selected_row["weighted_f1"]),
        threshold=float(selected_row["threshold"]) if pd.notna(selected_row["threshold"]) else None,
    )

    selected_train_features, selected_test_features = feature_spaces[selected_result.feature_space]
    selected_top_model_names = top_model_names_by_space[selected_result.feature_space]

    diagnostic_model = _fit_selected_model(selected_result, selected_top_model_names)
    diagnostic_model.fit(
        selected_train_features.iloc[train_indices].reset_index(drop=True),
        y.iloc[train_indices].reset_index(drop=True),
    )
    diagnostic_X_val = selected_train_features.iloc[val_indices].reset_index(drop=True)
    diagnostic_y_val = y.iloc[val_indices].reset_index(drop=True)
    if selected_result.threshold is None:
        diagnostic_predictions = diagnostic_model.predict(diagnostic_X_val)
    else:
        diagnostic_probabilities = diagnostic_model.predict_proba(diagnostic_X_val)[:, 1]
        diagnostic_predictions = (diagnostic_probabilities >= selected_result.threshold).astype(int)
    holdout_report = classification_report(diagnostic_y_val, diagnostic_predictions)

    final_model = _fit_selected_model(selected_result, selected_top_model_names)
    final_model.fit(selected_train_features, y)
    if selected_result.threshold is None:
        test_predictions = final_model.predict(selected_test_features)
    else:
        test_probabilities = final_model.predict_proba(selected_test_features)[:, 1]
        test_predictions = (test_probabilities >= selected_result.threshold).astype(int)

    submission_path = Path(submission_path)
    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": test_predictions.astype(int),
        }
    )
    submission.to_csv(submission_path, index=False)

    print("Bonus 2 improved workflow")
    print(leaderboard.to_string(index=False))
    print()
    print("Selected feature space:", selected_result.feature_space)
    print("Selected model:", selected_result.model_name)
    print(f"Validation Weighted F1: {selected_result.weighted_f1:.4f}")
    if selected_result.threshold is not None:
        print(f"Selected threshold: {selected_result.threshold:.2f}")
    print("Submission file:", submission_path)

    return {
        "leaderboard": leaderboard,
        "selected_result": selected_result,
        "holdout_report": holdout_report,
        "submission_path": submission_path,
    }
