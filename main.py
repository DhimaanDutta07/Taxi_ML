from src.data.ingest import DataLoader
from src.core.config import CFG
from src.features.build import FeatureBuilder
from src.data.clean import DataCleaner
from src.data.split import DataSplitter
from src.model.trainer import Trainer
from src.model.tune import Optunatuner
from src.explain.shap_report import ShapReport
from src.monitoring.evidently_report import EvidentlyReport


def main():
    loader = DataLoader(CFG["postgres_url"], "taxi_data")
    loader.upload_csv(CFG["raw_data"])
    df = loader.load("select * from taxi_data")

    builder = FeatureBuilder()
    df = builder.build(df)

    cleaner = DataCleaner(df)
    df = cleaner.clean()

    splitter = DataSplitter()
    x_train, x_test, y_train, y_test = splitter.split(df)

    tuner = Optunatuner(x_train, y_train, model_name="xgb", cv=30)
    tune_out = tuner.tune(n_trials=5)
    best_params = tune_out["best_params"]

    print("BEST RMSE (CV):", tune_out["best_rmse"])
    print("BEST PARAMS:", best_params)

    trainer = Trainer(x_train, "xgb", best_params)
    results = trainer.train(x_train, x_test, y_train, y_test)

    print(results)

    shap = ShapReport(CFG["model_path"])
    shap.get_report(x_test)
    print("shap report saved at:", CFG["shap_report_path"])

    report = EvidentlyReport(CFG["curr_path"], CFG["ref_path"])
    report.get_report()
    print("evidently report saved at:", CFG["evidently_path"])


if __name__ == "__main__":
    main()
