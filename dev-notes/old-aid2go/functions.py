#### Obtain Node Sampling Probability Based on Power Law Distribution of Node Degrees

# Characteristics of the Power Law distribution ($ P(x) \propto x^{-\alpha} $):

# 1. Scale-Invariance: Power law distributions look the same at any scale.
# 2. Heavy Tails: There is a non-negligible probability of very large values.
# 3. Few Large, Many Small: Most values are small, but a few are very large.

# Fit powerlaw and convert to probabilities for node sampling


def fit_powerlaw(values):
    """
    Fit a power-law distribution to the input values and return the fitted values.

    Parameters:
    values (list): A list of numerical values to fit the power-law distribution.

    Returns:
    list: A list of fitted values using the power-law distribution.
    """

    fit = powerlaw.Fit(values, discrete=True)
    alpha = fit.power_law.alpha  # exponent
    xmin = fit.power_law.xmin  # minimum value for wich the power-law holds
    fitted_values = [(x / xmin) ** (-alpha) for x in values]

    # Plot distributions
    print(f"Estimated alpha: {alpha}")
    print(f"Estimated xmin: {xmin}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    fit.plot_pdf(color="b", linestyle="-", ax=ax)
    fit.power_law.plot_pdf(color="r", linestyle="--", ax=ax)
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.legend(["Empirical Data", "Fitted Power Law"])
    plt.title("Power Law Fit to Data")
    plt.show()

    return fitted_values


def normalize_to_probabilities(values):
    """
    Normalize a list of values to probabilities that sum to 1.

    Parameters:
    values (list): A list of numerical values.

    Returns:
    list: A list of normalized probabilities.
    """
    total_sum = sum(values)
    probabilities = [val / total_sum for val in values]
    return probabilities


def convert_degrees_to_probabilities(degrees):
    """
    Convert degree values to probabilities using power-law fits.

    Parameters:
    degrees (dict): A dictionary where keys are GO term nodes and values are their degree values.

    Returns:
    dict: A dictionary where keys are GO term nodes and values are their normalized probabilities.
    """
    # Step 1: Extract degree values
    degree_values = list(degrees)

    # Step 2: Fit power-law distribution
    degree_fitted = fit_powerlaw(degree_values)

    # Step 3: Normalize fitted values to probabilities
    probabilities = normalize_to_probabilities(degree_fitted)

    # Step 4: Create the resulting dictionary
    nodes = list(range(len(degree_values)))
    prob_dict = dict(zip(nodes, probabilities))

    return prob_dict


# Plot training metrics


# Plot loss and accuracy (training results)
def plot_loss_and_accuracy(metrics, save_path):
    epochs = list(metrics["train_losses"].keys())

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, list(metrics["train_losses"].values()), label="Training Loss")
    plt.plot(epochs, list(metrics["val_losses"].values()), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, list(metrics["train_aurocs"].values()), label="Training AUROC")
    plt.plot(epochs, list(metrics["val_aurocs"].values()), label="Validation AUROC")
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title("Training and Validation AUROC")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path / "training_metrics.png", dpi=300)
    plt.show()


# Analyze IA and tokens of top GO terms
def analyze_go_terms(
    df, top_go_ids, information_accretion_col="ia", text_col="definition", top_n=25
):
    """
    Filters GO terms based on importance, visualizes distributions of 'ia', and computes TF-IDF of definitions.

    Parameters:
    - df (pd.DataFrame): DataFrame containing GO terms with columns for 'ia' and preprocessed definitions.
    - top_go_ids (list): List of top GO IDs considered important.
    - column_name (str): Column name for the importance attribute (default 'ia').
    - preprocessed_col (str): Column name for preprocessed text data.
    - top_n (int): Number of top terms to visualize with TF-IDF.

    Returns:
    - Tuple[pd.DataFrame, list]: DataFrame filtered to top GO terms and list of key top tokens that differ.
    """
    # Filter DataFrame to get top GO terms
    top_go_df = df[df["go_id"].isin(top_go_ids)]

    # Plotting distributions
    plt.figure(figsize=(12, 6))
    plt.hist(df[information_accretion_col], bins=30, alpha=0.5, label="All GO Terms")
    plt.scatter(
        top_go_df[information_accretion_col],
        np.zeros_like(top_go_df[information_accretion_col]),
        color="red",
        label="Top GO Terms",
    )
    plt.title("Distribution of 'ia' across all GO terms and top GO terms")
    plt.xlabel("ia")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Function to compute and plot TF-IDF
    def plot_tfidf(texts, label):
        vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, ngram_range=(1, 2))
        tf_idf_matrix = vectorizer.fit_transform(texts)
        max_scores = np.max(tf_idf_matrix, axis=0).toarray().flatten()
        sorted_indices = np.argsort(max_scores)[::-1][:top_n]
        sorted_features = vectorizer.get_feature_names_out()[sorted_indices]
        sorted_scores = max_scores[sorted_indices]

        plt.figure(figsize=(12, 8))
        bars = plt.barh(sorted_features, sorted_scores, color="green")
        plt.xlabel("Max TF-IDF Score")
        plt.title(f"Top {top_n} Terms with Highest TF-IDF Scores in {label}")
        plt.gca().invert_yaxis()
        plt.show()

        return sorted_features

    # Compute and plot TF-IDF for all GO terms and top GO terms
    all_tokens = plot_tfidf(df[text_col], "All GO Terms")
    top_tokens = plot_tfidf(top_go_df[text_col], "Top GO Terms")

    # Identify key top tokens that are different between the two sets
    key_top_tokens = list(set(top_tokens) - set(all_tokens))

    return top_go_df, key_top_tokens
