import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, make_scorer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import shap
import matplotlib.pyplot as plt
import logging
import time
import os
import joblib
import concurrent.futures
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from river import drift
from sklearn.linear_model import SGDClassifier
import requests
import gym
from gym import spaces

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv

    RL_INSTALLED = True
except ModuleNotFoundError:
    print(
        "Stable Baselines 3 not installed. Some functions will not work. Please run `pip install stable-baselines3[extra]`"
    )
    RL_INSTALLED = False

LOG_FILE = "system_logs.csv"
LOG_FILE = os.path.abspath(LOG_FILE)
FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "network_traffic_in",
    "network_traffic_out",
    "user_activity",
    "disk_io",
]
ANOMALY_DETECTION_MODEL = "IsolationForest"
CLASSIFICATION_MODEL = "GradientBoostingClassifier"
MODEL_DIR = "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_JOBS = -1
LOG_LEVEL = logging.INFO
THREAT_INTEL_API_KEY = "YOUR_THREAT_INTEL_API_KEY"
THREAT_INTEL_URL = "https://api.example.com/threatintel"
DATA_DRIFT_THRESHOLD = 0.7
CONCEPT_DRIFT_THRESHOLD = 0.8
RESPONSE_ACTIONS = {
    "block_ip": 0,
    "isolate_host": 1,
    "quarantine_file": 2,
    "no_action": 3,
}

def create_time_based_features(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["minute"] = df["timestamp"].dt.minute
    df.set_index("timestamp", inplace=True)
    for col in df.select_dtypes(include=np.number).columns:
        df[f"{col}_rate_of_change"] = df[col].diff().fillna(0)

    window_size = 5
    for col in df.select_dtypes(include=np.number).columns:
        df[f"{col}_rolling_mean"] = (
            df[col].rolling(window=window_size).mean().fillna(df[col].mean())
        )
        df[f"{col}_rolling_std"] = df[col].rolling(window=window_size).std().fillna(0)

    df.reset_index(inplace=True)
    return df


def create_statistical_features(df):
    # This is a placeholder and needs to be tailored to your data.
    return df


def load_and_preprocess_data(log_file, feature_cols):
    try:
        logging.info(f"Loading data from {log_file}...")
        data = pd.read_csv(log_file)
        if "timestamp" in data.columns:
            data = create_time_based_features(
                data.copy()
            )
        else:
            logging.warning("No timestamp column found. Skipping time-based feature creation.")
        data = data.fillna(data.mean())
        data = data[feature_cols]
        imputer = KNNImputer(n_neighbors=5)
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        scaler = StandardScaler()
        data[feature_cols] = scaler.fit_transform(data[feature_cols])

        quantile_transformer = QuantileTransformer(
            output_distribution="normal", random_state=RANDOM_STATE
        )
        data[feature_cols] = quantile_transformer.fit_transform(data[feature_cols])

        logging.info("Data loading and preprocessing complete.")
        return data
    except FileNotFoundError:
        logging.error(f"Error: Log file not found: {log_file}")
        return None
    except KeyError as e:
        logging.error(
            f"Error: Missing columns in log file. Check column names and feature_cols: {e}"
        )
        return None
    except Exception as e:
        logging.exception(
            f"An unexpected error occurred during data loading/processing: {e}"
        )
        return None


def train_anomaly_detection_model(data, model_name, random_state=RANDOM_STATE):
    try:
        logging.info(f"Training anomaly detection model: {model_name}")
        if model_name == "IsolationForest":
            model = IsolationForest(
                contamination="auto", random_state=random_state, n_estimators=150, n_jobs=N_JOBS
            )
        elif model_name == "HBOS":
            from pyod.models.hbos import HBOS

            model = HBOS(contamination="auto")
        elif model_name == "AutoEncoder":
            from pyod.models.auto_encoder import AutoEncoder


            model = AutoEncoder(
                epochs=50, hidden_neurons=[64, 32, 64], random_state=random_state
            )

        else:
            logging.error(f"Model not implemented: {model_name}")
            return None

        cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=random_state
        )
        scores = cross_val_score(
            model, data, cv=cv, scoring="roc_auc", n_jobs=N_JOBS
        )
        logging.info(f"Cross-validation ROC AUC scores: {scores}")
        logging.info(f"Mean cross-validation ROC AUC score: {scores.mean()}")

        start_time = time.time()
        model.fit(data)
        end_time = time.time()
        logging.info(f"Model trained in {end_time - start_time:.2f} seconds")

        model_filename = os.path.join(MODEL_DIR, f"{model_name}_anomaly_model.joblib")
        joblib.dump(model, model_filename)
        logging.info(f"Model saved to {model_filename}")

        return model
    except Exception as e:
        logging.exception(f"Error training anomaly detection model: {e}")
        return None


def detect_anomalies(model, data):
    try:
        logging.info("Detecting anomalies...")
        predictions = model.predict(data)
        scores = model.decision_function(data)

        results = pd.DataFrame({"prediction": predictions, "anomaly_score": scores})
        logging.info("Anomaly detection complete.")
        return results
    except Exception as e:
        logging.exception(f"Error detecting anomalies: {e}")
        return None


def train_threat_classification_model(
    X_train, y_train, model_name=CLASSIFICATION_MODEL, random_state=RANDOM_STATE
):
    try:
        logging.info(f"Training threat classification model: {model_name}")
        if model_name == "RandomForestClassifier":
            model = RandomForestClassifier(
                random_state=random_state, n_estimators=100, class_weight="balanced"
            )
        elif model_name == "GradientBoostingClassifier":
            model = GradientBoostingClassifier(
                random_state=random_state, n_estimators=100, learning_rate=0.1, max_depth=5
            )
        else:
            logging.error(f"Model not implemented: {model_name}")
            return None

        cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=random_state
        )
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=N_JOBS
        )
        logging.info(f"Cross-validation F1 scores: {scores}")
        logging.info(f"Mean cross-validation F1 score: {scores.mean()}")

        model.fit(X_train, y_train)

        model_filename = os.path.join(MODEL_DIR, f"{model_name}_threat_model.joblib")
        joblib.dump(model, model_filename)
        logging.info(f"Model saved to {model_filename}")

        return model
    except Exception as e:
        logging.exception(f"Error training classification model: {e}")
        return None


def classify_threat(model, data):
    """Classifies threats using the trained model."""
    try:
        logging.info("Classifying threats...")
        predictions = model.predict(data)
        logging.info("Threat classification complete.")
        return predictions
    except Exception as e:
        logging.exception(f"Error classifying threats: {e}")
        return None


def explain_anomalies(model, data, anomaly_indices, top_n_features=5):
    try:
        logging.info("Explaining anomalies using SHAP...")

        background_data = data.iloc[
            np.random.choice(data.index, size=min(100, len(data)), replace=False)
        ]
        explainer = shap.KernelExplainer(model.predict, background_data)

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            shap_values = list(
                executor.map(explainer.shap_values, [data.iloc[[i]] for i in anomaly_indices])
            )
            shap_values = np.array(shap_values).squeeze()

        num_anomalies_to_explain = min(5, len(anomaly_indices))
        for i in range(num_anomalies_to_explain):
            anomaly_index = anomaly_indices[i]
            shap.force_plot(
                explainer.expected_value,
                shap_values[i],
                data.iloc[anomaly_index],
                show=False,
                link="logit",
            )
            plt.title(f"SHAP Force Plot for Anomaly {i + 1} (Index: {anomaly_index})")
            plt.show()

        feature_importances = np.abs(shap_values).mean(axis=0)
        top_feature_indices = np.argsort(feature_importances)[::-1][
            :top_n_features
        ]
        top_feature_names = data.columns[top_feature_indices]
        logging.info(
            f"Top {top_n_features} important features for anomaly detection: {list(top_feature_names)}"
        )

        shap.summary_plot(
            shap_values, data, feature_names=data.columns, max_display=top_n_features, show=False
        )
        plt.title("SHAP Summary Plot for Anomaly Explanations (Top Features)")
        plt.show()

        logging.info("Anomaly explanation complete.")
        return shap_values
    except Exception as e:
        logging.exception(f"Error explaining anomalies: {e}")
        return None


def enrich_with_threat_intel(anomaly_data):
    try:
        ip_address = anomaly_data.get("network_traffic_in")
        if ip_address:
            headers = {"Authorization": f"Bearer {THREAT_INTEL_API_KEY}"}
            response = requests.get(f"{THREAT_INTEL_URL}?ip={ip_address}", headers=headers)
            response.raise_for_status()
            intel_data = response.json()
            logging.info(f"Threat intel for IP {ip_address}: {intel_data}")
            anomaly_data["threat_score"] = intel_data.get("threat_score", 0)
            anomaly_data["is_malicious"] = intel_data.get("is_malicious", False)
        else:
            logging.warning("No IP address found in anomaly data")

        return anomaly_data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying threat intel API: {e}")
        return anomaly_data


class ThreatResponseEnv(gym.Env):

    def __init__(self, initial_state=None):
        super(ThreatResponseEnv, self).__init__()
        self.action_space = spaces.Discrete(len(RESPONSE_ACTIONS))
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(7,), dtype=np.float32
        )
        self.initial_state = initial_state

    def step(self, action):
        action_name = list(RESPONSE_ACTIONS.keys())[action]
        logging.info(f"Executing action: {action_name}")

        reward = self._calculate_reward(action)

        done = True

        self.state = self._get_next_state()

        info = {}
        return self.state, reward, done, info

    def reset(self):
        self.state = self.initial_state
        return self.state

    def _calculate_reward(self, action):
        if action == RESPONSE_ACTIONS["no_action"]:
            return -1
        elif action == RESPONSE_ACTIONS["block_ip"]:
            return 5
        elif action == RESPONSE_ACTIONS["isolate_host"]:
            return 7
        else:
            return -2

    def _get_next_state(self):
        return self.state


def trigger_adaptive_threat_response(anomaly_data, env, model):
    try:
        state = np.array(
            [
                anomaly_data.get("cpu_usage", 0),
                anomaly_data.get("memory_usage", 0),
                anomaly_data.get("network_traffic_in", 0),
                anomaly_data.get("network_traffic_out", 0),
                anomaly_data.get("user_activity", 0),
                anomaly_data.get("disk_io", 0),
                anomaly_data.get("threat_score", 0),
            ],
            dtype=np.float32,
        )

        env.initial_state = state
        state = env.reset()

        action = model.predict(state.reshape(1, -1))[0]
        new_state, reward, done, info = env.step(action)

        logging.info("Chose Action {}".format(action))
        logging.info("The reward received from the action is {}".format(reward))

        return action

    except Exception as e:
        logging.exception(f"Error triggering adaptive threat response: {e}")
        return RESPONSE_ACTIONS["no_action"]


def detect_data_drift(reference_data, current_data, column_mapping=None):
    try:
        report = Report(metrics=[DataDriftPreset()])

        if column_mapping is None:
            column_mapping = ColumnMapping()
        report.run(
            reference_data=reference_data, current_data=current_data, column_mapping=column_mapping
        )
        report.show()
        results = report.as_dict()
        drift_share = results["metrics"][0]["result"]["drift_share"]
        logging.info(f"Data Drift Share: {drift_share}")
        return drift_share
    except Exception as e:
        logging.exception(f"Error detecting data drift: {e}")
        return 1.0


def train_online_classifier(initial_X, initial_y, model_name="SGDClassifier"):
    try:
        if model_name == "SGDClassifier":
            model = SGDClassifier(
                loss="log_loss", random_state=RANDOM_STATE
            )

        else:
            logging.error(f"Online model not implemented: {model_name}")
            return None

        model.fit(initial_X, initial_y)
        return model

    except Exception as e:
        logging.exception(f"Error training online classifier: {e}")
        return None


def detect_concept_drift(online_model, new_X, new_y):
    try:
        adwin = drift.ADWIN(delta=0.002)

        predictions = online_model.predict(new_X)
        for i, (xi, yi, prediction) in enumerate(zip(new_X.values, new_y, predictions)):
            online_model = online_model.partial_fit([xi], [yi], classes=np.unique(new_y))
            adwin.update(int(prediction != yi))

            if adwin.drift_detected():
                logging.warning("Concept Drift Detected")
                return True

        return False

    except Exception as e:
        logging.exception(f"Error detecting concept drift: {e}")
        return True


def verify_log_file(log_file):
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at the specified path: {log_file}")
        return False
    if not os.path.isfile(log_file):
        print(f"Error: The specified path is not a file: {log_file}")
        return False
    if not os.access(log_file, os.R_OK):
        print(f"Error: Insufficient permissions to read the log file: {log_file}")
        return False
    return True


def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting Advanced AI-Powered Threat Hunting...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    start_time = time.time()

    if not verify_log_file(LOG_FILE):
        logging.error("Log file verification failed. Exiting.")
        return

    initial_data = load_and_preprocess_data(LOG_FILE, FEATURE_COLUMNS)
    end_time = time.time()

    if initial_data is None:
        logging.error("Exiting due to data loading or preprocessing errors.")
        return
    logging.info(f"Initial data loading and preprocessing took {end_time - start_time:.2f} seconds")

    train_data, reference_data = train_test_split(
        initial_data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    start_time = time.time()
    anomaly_model = train_anomaly_detection_model(train_data, ANOMALY_DETECTION_MODEL)
    end_time = time.time()
    if anomaly_model is None:
        logging.error("Exiting due to anomaly detection model training errors.")
        return
    logging.info(f"Anomaly detection model training took {end_time - start_time:.2f} seconds")

    start_time = time.time()
    anomaly_results = detect_anomalies(anomaly_model, reference_data)
    end_time = time.time()
    if anomaly_results is None:
        logging.error("Exiting due to anomaly detection errors.")
        return
    logging.info(f"Anomaly detection took {end_time - start_time:.2f} seconds")

    anomaly_indices = anomaly_results[anomaly_results["prediction"] == -1].index.tolist()
    logging.info(f"Number of anomalies detected: {len(anomaly_indices)}")

    try:
        threat_labels = pd.read_csv("threat_labels.csv", index_col="index").to_dict()[
            "threat_type"
        ]
        X_anomalies = train_data.iloc[
            [index for index in threat_labels.keys() if index in train_data.index]
        ]
        y_anomalies = [
            threat_labels[index] for index in threat_labels.keys() if index in train_data.index
        ]

        X_train, X_test, y_train, y_test = train_test_split(
            X_anomalies,
            y_anomalies,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_anomalies,
        )
        X_test = X_test.reset_index(drop=True)
        y_test = pd.Series(y_test).reset_index(drop=True)

        online_model = train_online_classifier(X_train, y_train)

        concept_drift = detect_concept_drift(online_model, X_test, y_test)

        if concept_drift:
            online_model = train_online_classifier(
                X_train, y_train
            )
            print("retraining online model")

        if online_model:
            predictions = online_model.predict(X_test)
            print("Threat Classification Report:")
            print(classification_report(y_test, predictions))

    except FileNotFoundError:
        logging.error("Exiting Threat Classification - threat_labels.csv not found")
    except Exception as e:
        logging.error("Error during threat Classification{}".format(e))

    if len(anomaly_indices) > 0:
        start_time = time.time()
        explain_anomalies(anomaly_model, reference_data, anomaly_indices)
        end_time = time.time()
        logging.info(f"Anomaly explanation took {end_time - start_time:.2f} seconds")

    start_time = time.time()
    if RL_INSTALLED:
        try:

            env = ThreatResponseEnv()
            rl_model = DQN.load("dqn_threat_response", env=env)
        except Exception as e:
            logging.warning(f"RL environment and agent not initialized. Action is skipped. Error: {e}")
            rl_model = None


        for index in anomaly_indices:
            enriched_data = enrich_with_threat_intel(
                reference_data.iloc[index].to_dict()
            )
            if "env" in locals() and rl_model is not None:
                action = trigger_adaptive_threat_response(enriched_data, env, rl_model)
                logging.warning(
                    "Action Triggered: {} for {}".format(
                        list(RESPONSE_ACTIONS.keys())[action], enriched_data
                    )
                )
            else:
                logging.warning("Adaptive Response Skipped. Env or RL Agent missing")
    else:
        logging.warning("Adaptive Response Skipped. stable_baselines3 not installed.")

    end_time = time.time()
    logging.info(f"Threat response took {end_time - start_time:.2f} seconds")

    drift_score = detect_data_drift(reference_data, initial_data)
    if drift_score > DATA_DRIFT_THRESHOLD:
        logging.warning("Significant data drift detected. Retraining anomaly detection model.")

        combined_data = pd.concat([reference_data, initial_data])
        anomaly_model = train_anomaly_detection_model(combined_data, ANOMALY_DETECTION_MODEL)
        reference_data = initial_data

    logging.info("Advanced Threat Hunting Complete.")


if __name__ == "__main__":
    main()