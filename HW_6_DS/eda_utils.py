import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")


def plot_top_distributions(df, n=8):

    numeric = df.select_dtypes(include=["int64","float64"]).columns[:n]

    fig, axes = plt.subplots(2,4, figsize=(16,8))

    axes = axes.flatten()

    for i,col in enumerate(numeric):

        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)

    plt.tight_layout()
    plt.show()



def plot_correlation(df):

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12,10))

    sns.heatmap(corr, cmap="coolwarm", center=0)

    plt.title("Correlation Matrix")

    plt.show()



def pairplot_features(df):

    numeric = df.select_dtypes(include=["int64","float64"]).columns[:6]

    sns.pairplot(df[numeric])

    plt.show()



def pca_visualization(df):

    numeric = df.select_dtypes(include=["int64","float64"])

    scaler = StandardScaler()

    scaled = scaler.fit_transform(numeric)

    pca = PCA(n_components=2)

    components = pca.fit_transform(scaled)

    plt.figure(figsize=(8,6))

    plt.scatter(
        components[:,0],
        components[:,1],
        alpha=0.4
    )

    plt.title("PCA Projection")

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.show()