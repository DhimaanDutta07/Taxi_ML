import os
import joblib
import shap
import matplotlib.pyplot as plt
from src.core.config import CFG

class ShapReport():
    def __init__(self,model_path):
        self.model_path=CFG["model_path"]
        self.pipe=joblib.load(self.model_path)
        self.pre=self.pipe.named_steps["preprocessor"]
        self.model=self.pipe.named_steps["model"]
    
    def get_report(self,x_test):
        x_test_transformed=self.pre.transform(x_test)
        explainer=shap.TreeExplainer(self.model)
        shap_values=explainer.shap_values(x_test_transformed)
        shap.summary_plot(shap_values, x_test_transformed, show=False)
        plt.savefig(CFG["shap_report_path"], bbox_inches="tight", dpi=250)
        plt.close()

