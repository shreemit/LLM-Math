import pandas as pd
import argparse

def evaluate_results(df):
    # Define the categories for number of digits and operators
    num_digits_categories = ["Three digits", "Four digits", "Five digits"]
    operator_categories = ["addition", "subtraction", "multiplication", "division"]

    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=["Category", "Accuracy"])

    # Evaluation of the results for number of digits
    for num_digits in num_digits_categories:
        subset = zeroshot_results[zeroshot_results['Num_Digits'] == num_digits]
        accuracy = subset.Correct.sum() / len(subset)
        results_df.loc[len(results_df)] = [num_digits, accuracy]
        results_df.index = range(len(results_df))
        results_df = results_df.sort_index()

    # Evaluation of the results for operators
    for operator in operator_categories:
        subset = zeroshot_results[zeroshot_results['Operator'] == operator]
        accuracy = subset.Correct.sum() / len(subset)
        # results_df = results_df.append({"Category": operator, "Accuracy": accuracy}, ignore_index=True)
        results_df.loc[len(results_df)] = [operator, accuracy]
        results_df.index = range(len(results_df))
        results_df = results_df.sort_index()

    # Round off the answer to 2 decimal places
    zeroshot_results['Answer'] = zeroshot_results['Answer'].apply(lambda x: round(float(x), 2))

    # Get accuracy of the results comparing answer and predicted
    zeroshot_results['Correct'] = zeroshot_results.apply(lambda x: 1 if round(float(x['Predicted']), 2) == round(float(x['Answer']), 2) else 0, axis=1)
    accuracy = zeroshot_results['Correct'].sum() / len(zeroshot_results)
    # results_df = results_df.append({"Category": "Overall", "Accuracy": accuracy}, ignore_index=True)
    results_df.loc[len(results_df)] = ["Overall", accuracy]
    results_df.index = range(len(results_df))
    results_df = results_df.sort_index()

    print(results_df)

    # Get accuracy of the results comparing answer and predicted
    df_copy = df.copy()
    df_copy['Correct'] = df_copy.apply(lambda x: 1 if round(float(x['Predicted']), 2) == round(float(x['Answer']), 2) else 0, axis=1)
    accuracy = df_copy['Correct'].sum() / len(df)
    print(f"Overall accuracy (rounded to 2 decimal places): {accuracy}")

# Create the parser
parser = argparse.ArgumentParser(description="Evaluate results")

# Add the arguments
parser.add_argument("ResultsPath", metavar="path", type=str, help="The path to the results CSV file")

# Parse the arguments
args = parser.parse_args()

# Load the DataFrame from the CSV file
zeroshot_results = pd.read_csv(args.ResultsPath)

# Call the evaluation function
evaluate_results(zeroshot_results)