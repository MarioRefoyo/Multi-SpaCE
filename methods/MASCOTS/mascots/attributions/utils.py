import shap

shap_mapping = {
    "normal": shap.Explainer,
    "linear": shap.LinearExplainer,
    "tree": shap.TreeExplainer,
    "kernel": shap.KernelExplainer,
    "gradient": shap.GradientExplainer,
    "deep": shap.DeepExplainer,
}
