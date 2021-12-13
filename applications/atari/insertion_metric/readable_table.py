""" Module for cleaning up the cluttered results produced by insertion metric plot
    This includes combining mean and SD with a plus-minus sign and formatting it for latex.
"""

import pandas as pd
import os

DECIMAL_PLACES = 2

GAMES = ["pacman", "breakout", "frostbite", "spaceInvaders"]
INSERTION_COLORS = ["random_insertion", "black_insertion"]

if __name__ == '__main__':
    for game in GAMES:
        df = pd.read_csv(os.path.join("results", game + "_mean_AUCS.csv"))
        resulting_df = pd.DataFrame()
        resulting_df["Approach"] = df["Approach"]
        for INSERTION_COLOR in INSERTION_COLORS:
            for ADVANTAGE in [0,1]:
                mean_aucs = df["mean_auc_" + game + "_" + str(ADVANTAGE) + "_" + INSERTION_COLOR].values
                stds = df["std_" + game + "_" + str(ADVANTAGE) + "_" + INSERTION_COLOR].values

                combined_values = [f"{round(mean_auc, DECIMAL_PLACES)}$\pm${round(std, DECIMAL_PLACES-1)}" for mean_auc, std in zip(mean_aucs, stds)]
                if ADVANTAGE:
                    measure = "advantage"
                else:
                    measure = "qvals"
                column_name = f"{INSERTION_COLOR}_{measure}"
                resulting_df[column_name] = combined_values

        resulting_df = resulting_df.T
        resulting_df.to_csv(os.path.join("results",game + "_cleaned.csv"))

        with open(os.path.join("results",f"{game}_latex_table.txt"), 'w') as output_file:
            output_file.write(resulting_df.to_latex(index=True, escape=False))