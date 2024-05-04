import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns




def compute_team_average_stats(game_data_path, team_details_path, output_path):
    df = pd.read_csv(game_data_path)
    teams_df = pd.read_csv(team_details_path)

    features_home = ['fg3_pct_home', 'ft_pct_home', 'oreb_home', 'dreb_home', 'reb_home', 'ast_home', 'stl_home',
                     'blk_home', 'tov_home', 'wl_home']
    features_away = ['fg3_pct_away', 'ft_pct_away', 'oreb_away', 'dreb_away', 'reb_away', 'ast_away', 'stl_away',
                     'blk_away', 'tov_away', 'wl_away']

    home_stats = df[['team_id_home'] + features_home]
    away_stats = df[['team_id_away'] + features_away]

    home_stats.columns = ['team_id'] + [col.replace('_home', '') for col in features_home]
    away_stats.columns = ['team_id'] + [col.replace('_away', '') for col in features_away]

    combined_stats = pd.concat([home_stats, away_stats], ignore_index=True)
    average_stats = combined_stats.groupby('team_id').agg(lambda x: x.mean()).reset_index()
    average_stats.columns = [col if col != 'team_id' else col for col in average_stats.columns]

    average_stats = pd.merge(average_stats, teams_df[['team_id', 'abbreviation']], on='team_id')

    average_stats.to_csv(output_path, index=False)
    print(f"Team average stats saved to {output_path}")


def logistic_regression_model(x, y, description, printing, plot=False):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{description} Logistic Regression Model Accuracy: {accuracy:.2f}')
    # cm = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix:")
    # print(cm)
    coefficients = model.coef_[0]
    feature_names = x.columns.tolist()

    sorted_indices = sorted(range(len(coefficients)), key=lambda i: abs(coefficients[i]), reverse=True)
    sorted_coefficients = [coefficients[i] for i in sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    if printing == 1:
        print("WEIGHTS:")
        for coefficient, feature_name in zip(sorted_coefficients, sorted_feature_names):
            print(f"{feature_name}: {coefficient}")
    if plot:
        matplotlib.use('TkAgg')

        coef_df = pd.DataFrame({'Feature': sorted_feature_names, 'Coefficient': sorted_coefficients})
        plt.figure(figsize=(10, 8))
        sns.barplot(x="Coefficient", y="Feature", data=coef_df)
        plt.title('Feature Importances in Logistic Regression')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Features')
        plt.show()

    return model


def augment_game_data_with_team_stats(game_data_path, team_stats_path):
    df_games = pd.read_csv(game_data_path)
    df_team_avg = pd.read_csv(team_stats_path)
    home_columns = {col: col + "_home_avg" for col in df_team_avg.columns if col not in ["team_id", "abbreviation"]}
    away_columns = {col: col + "_away_avg" for col in df_team_avg.columns if col not in ["team_id", "abbreviation"]}

    df_team_avg_home = df_team_avg.rename(columns=home_columns)
    df_team_avg_away = df_team_avg.rename(columns=away_columns)

    df_merged = pd.merge(df_games, df_team_avg_home, how="left", left_on="team_id_home", right_on="team_id")
    df_merged = pd.merge(df_merged, df_team_avg_away, how="left", left_on="team_id_away", right_on="team_id")
    df = df_merged.dropna(axis=0)

    df.to_csv('data/merged_data_complete.csv', index=False)
    # print(df)
    return df

def svm_model(X, y, description, printing):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = SVC(kernel='linear', random_state=42)  # Linear kernel for simplicity, can try other kernels
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    print(f'{description} SVM Model Accuracy: {accuracy:.2f}')

    if printing == 1:
        print("SVM Parameters:")
        print(model.get_params())

    return model

def neural_network_model(x, y, description):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    print(f'{description} Neural Network Model Accuracy: {accuracy:.2f}')

    return model


def train_and_visualize_decision_tree(data_path, features, target, test_size=0.3, random_state=42, max_depth=5):
    matplotlib.use('TkAgg')
    df = pd.read_csv(data_path)

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    plt.figure(figsize=(20, 10))
    plot_tree(tree, filled=True, feature_names=features, class_names=['Loss', 'Win'], rounded=True, fontsize=12)
    plt.show()

def main():
    game_data_path = 'data/game_data.csv'
    team_details_path = 'data/team_details.csv'
    combined_stats_path = 'data/team_average_stats_combined.csv'

    compute_team_average_stats(game_data_path, team_details_path, combined_stats_path)

    original_df = pd.read_csv(game_data_path)
    team_stats_path = 'data/team_average_stats_combined.csv'

    features_home = ['fg3_pct_home', 'ft_pct_home', 'oreb_home', 'dreb_home', 'reb_home', 'ast_home', 'stl_home',
                     'blk_home',
                     'tov_home']
    target_home = 'wl_home'

    x_original_home = original_df[features_home]
    y_original_home = original_df[target_home]

    logistic_regression_model(x_original_home, y_original_home, "Home Game Biased-", 0)

    # AWAY
    features_away = ['fg3_pct_away', 'ft_pct_away', 'oreb_away', 'dreb_away', 'reb_away', 'ast_away', 'stl_away',
                     'blk_away',
                     'tov_away']
    target_away = 'wl_away'

    x_original_away = original_df[features_away]
    y_original_away = original_df[target_away]

    # ACCURACY HERE WILL BE TOO HIGH DUE TO OVERFIT
    logistic_regression_model(x_original_away, y_original_away, "Away Game Biased-", 0)
    augmented_game_data = augment_game_data_with_team_stats(game_data_path, team_stats_path)

    features_combined = ['fg3_pct_home_avg', 'ft_pct_home_avg', 'oreb_home_avg', 'dreb_home_avg', 'ast_home_avg',
                         'reb_home_avg', 'stl_home_avg', 'blk_home_avg', 'tov_home_avg', 'fg3_pct_away_avg', 'ft_pct_away_avg', 'oreb_away_avg', 'dreb_away_avg', 'ast_away_avg',
                         'reb_away_avg', 'stl_away_avg', 'blk_away_avg', 'tov_away_avg']

    features_baseline = ['fg3_pct_home_avg', 'fg3_pct_away_avg']

    x_combined = augmented_game_data[features_combined]
    x_baseline = augmented_game_data[features_baseline]
    y_combined = augmented_game_data['wl_home']
    neural_network_model(x_combined, y_combined, "Season Averages-")

    logistic_regression_model(x_baseline, y_combined, "Baseline Season Averages-", 0)
    logistic_regression_model(x_combined, y_combined, "Improved Season Averages-", 1, True)
    train_and_visualize_decision_tree('data/merged_data_complete.csv', features_combined, 'wl_home')


if __name__ == "__main__":
    main()
