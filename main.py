import rich

import utils
from model import PremiumModel
from preprocess import X_freq, X_sev, df, sev_mask, w_freq, w_sev, y_freq, y_sev

def run_model_cv(cls: type[PremiumModel], args: tuple = (), kwargs: dict = {}, save_pattern: str | None = "{model_name}_fold_{fold}.pkl") -> dict[str, list]:
    """
    Run cross-validation for a premium model.

    Parameters:
    -----------
    model_fn : Callable that returns a PremiumModel instance

    Returns:
    --------
    Dictionary with cross-validation results
    """
    results = {
        "fold": [],
        "freq_deviance": [],
        "sev_deviance": [],
        "freq_likelihood": [],
        "sev_likelihood": [],
        "price_ratios": [],
    }

    rich.print("Starting cross-validation...")
    rich.print(f"Model: '{cls.__name__}'")
    rich.print(f"Args: {args}")
    rich.print(f"Kwargs: {kwargs}")

    for fold in sorted(df["fold"].unique()):
        print(f"Processing fold {fold}...")

        train_mask = df["fold"] != fold
        test_mask = df["fold"] == fold
        sev_train_mask = train_mask & sev_mask
        sev_test_mask = test_mask & sev_mask

        # Initialize and fit the model
        model = cls(*args, **kwargs)
        model.fit(X_freq[train_mask], y_freq[train_mask], w_freq[train_mask], X_sev[sev_train_mask], y_sev[sev_train_mask], w_sev[sev_train_mask])

        # Save the trained model
        if save_pattern is not None:
            model.save(save_pattern.format(model_name=cls.__name__, fold=fold))

        # Predict
        freq_pred, sev_pred = model.predict(X_freq[test_mask], X_sev[test_mask])

        # Evaluate performance
        results["fold"].append(fold)
        results["freq_deviance"].append(utils.mean_poisson_deviance(y_freq[test_mask], freq_pred, sample_weight=w_freq[test_mask]))
        results["freq_likelihood"].append(utils.mean_poisson_loglike_full(y_freq[test_mask], freq_pred, sample_weight=w_freq[test_mask]))
        results["sev_deviance"].append(utils.mean_gamma_deviance(y_sev[sev_test_mask], sev_pred[sev_mask[test_mask]], sample_weight=w_sev[sev_test_mask]))
        results["sev_likelihood"].append(utils.mean_gamma_loglike_full(y_sev[sev_test_mask], sev_pred[sev_mask[test_mask]], sample_weight=w_sev[sev_test_mask]))
        results["price_ratios"].append((freq_pred * sev_pred * w_freq[test_mask]).sum() / df["amount"][test_mask].sum())

        print("price ratios:", results["price_ratios"][-1])

    return results


if __name__ == "__main__":
    # from model import GLMPremiumModel
    # results = run_model_cv(GLMPremiumModel)

    # from model import GAMPremiumModel
    # results = run_model_cv(
    #     GAMPremiumModel,
    #     kwargs=dict(
    #         frequency_terms=["coverage", "fuel", "ageph", "power", "bm", ("ageph", "power"), ("long", "lat")],
    #         severity_terms=["coverage", "ageph", "bm"],
    #     ),
    # )

    # from model import GBMPremiumModel
    # results = run_model_cv(
    #     GBMPremiumModel,
    #     kwargs=dict(
    #         frequency_params=dict(
    #             max_iter=100,  # Number of boosting stages
    #             learning_rate=0.05,  # Learning rate
    #             max_bins=2,
    #             max_depth=1,  # Maximum depth of trees
    #             random_state=42,
    #         ),
    #         severity_params=dict(
    #             max_iter=100,  # Number of boosting stages
    #             learning_rate=0.2,  # Learning rate
    #             max_bins=2,
    #             max_depth=1,  # Slightly shallower for severity
    #             random_state=42,  # For reproducibility
    #         ),
    #     ),
    # )

    # from model import FFNNPremiumModel
    # results = run_model_cv(
    #     FFNNPremiumModel,
    #     args=(),
    #     kwargs=dict(
    #         frequency_params=dict(
    #             hidden_dims=[50, 25],
    #             dropout_rate=0.2,
    #         ),
    #         severity_params=dict(
    #             hidden_dims=[50, 25],
    #             dropout_rate=0.2,
    #         ),
    #         training_params=dict(
    #             batch_size=128,
    #             learning_rate=0.001,
    #             weight_decay=0.0001,
    #             patience=30,
    #             validation_split=0.2,
    #             num_epochs=1000,
    #             verbose=True,
    #             log_interval=20,
    #             seed=42,
    #         ),
    #     ),
    # )

    # from model import MTNNPremiumModel
    # results = run_model_cv(
    #     MTNNPremiumModel, 
    #     kwargs=dict(
    #         shared_encoder_params=dict(
    #             hidden_dims=[50, 25],
    #             dropout_rate=0.3,
    #         ),
    #         freq_decoder_params=dict(
    #             hidden_dims=[20, 10],
    #             dropout_rate=0.1,
    #         ),
    #         sev_decoder_params=dict(
    #             hidden_dims=[25, 10],
    #             dropout_rate=0.1,
    #         ),
    #         training_params=dict(
    #             batch_size=128,
    #             learning_rate=0.001,
    #             weight_decay=0.0006,
    #             patience=30,
    #             validation_split=0.2,
    #             num_epochs=1000,
    #             verbose=True,
    #             log_interval=20,
    #             seed=42,
    #             freq_weight=3.0,
    #             sev_weight=0.2,
    #         ),
    #     ),
    # )

    # from model import MTNNPremiumModel
    # results = run_model_cv(
    #     MTNNPremiumModel, 
    #     kwargs=dict(
    #         shared_encoder_params=dict(
    #             hidden_dims=[20, 40],
    #             dropout_rate=0.1,
    #         ),
    #         freq_decoder_params=dict(
    #             hidden_dims=[10],
    #             dropout_rate=0,
    #         ),
    #         sev_decoder_params=dict(
    #             hidden_dims=[20, 10],
    #             dropout_rate=0.1,
    #         ),
    #         training_params=dict(
    #             batch_size=128,
    #             learning_rate=0.001,
    #             weight_decay=0.0006,
    #             patience=30,
    #             validation_split=0.2,
    #             num_epochs=1000,
    #             verbose=True,
    #             log_interval=20,
    #             seed=42,
    #             freq_weight=3.0,
    #             sev_weight=0.2,
    #         ),
    #         device="cuda:1",
    #     ),
    #     save_pattern="{model_name}_small_fold_{fold}.pkl"
    # )

    from model import MTMoENNPremiumModel
    results = run_model_cv(
        MTMoENNPremiumModel,
        kwargs=dict(
            shared_encoder_params=dict(
                hidden_dims=[50, 25],
                dropout_rate=0.3,
            ),
            freq_decoder_params=dict(
                hidden_dims=[20, 10],
                dropout_rate=0.1,
            ),
            sev_decoders_params=dict(
                hidden_dims=[[25, 10], [25, 10]],
                dropout_rate=0.1,
            ),
            training_params=dict(
                batch_size=128,
                learning_rate=0.001,
                weight_decay=0.0005,
                patience=30,
                validation_split=0.2,
                num_epochs=1000,
                verbose=True,
                log_interval=20,
                seed=42,
                freq_weight=3.0,
                sev_weight=0.2,
            ),
            device="cuda:1",
        ),
        save_pattern="{model_name}_small_fold_{fold}.pkl"
    )
    

    import pandas as pd
    results_df = pd.DataFrame(results)
    print(results_df.set_index("fold"))
    print("Mean Price Ratio:", results_df["price_ratios"].mean())
    print("Price Ratio Std Dev:", results_df["price_ratios"].std())
