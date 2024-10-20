import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    transactions = data.groupby(['Invoice ID', 'Product line'])['Product line'].count().unstack().reset_index().fillna(0)
    transactions = transactions.drop('Invoice ID', axis=1)
    transactions = transactions.applymap(lambda x: 1 if x > 0 else 0)
    return transactions

def run_apriori(transactions, min_support=0.01):
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    return frequent_itemsets, rules

def run_fp_growth(transactions, min_support=0.01):
    frequent_itemsets = fpgrowth(transactions, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    return frequent_itemsets, rules

def format_output(frequent_itemsets, rules):
    output = "Table 1: Frequent Itemsets with Support\n"
    output += "{:<40} {:<10}\n".format("Itemset", "Support")
    output += "-" * 50 + "\n"
    for _, row in frequent_itemsets.iterrows():
        output += "{:<40} {:<10.2%}\n".format(", ".join(row['itemsets']), row['support'])
    
    output += "\nTable 2: Association Rules\n"
    output += "{:<40} {:<40} {:<10} {:<10}\n".format("Antecedent", "Consequent", "Support", "Confidence")
    output += "-" * 100 + "\n"
    for _, row in rules.iterrows():
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        output += "{:<40} {:<40} {:<10.2%} {:<10.2%}\n".format(antecedents, consequents, row['support'], row['confidence'])
    
    return output

def display_output(output):
    output_window = tk.Toplevel(root)
    output_window.title("Analysis Output")
    
    text_widget = tk.Text(output_window, wrap="none", width=100, height=30)
    text_widget.pack(side="left", fill="both", expand=True)
    
    scrollbar = tk.Scrollbar(output_window, command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")
    
    text_widget.config(yscrollcommand=scrollbar.set)
    text_widget.insert(tk.END, output)
    
    text_widget.config(state=tk.DISABLED)

def run_algorithm():
    csv_file = file_entry.get()
    if not csv_file:
        messagebox.showwarning("Input Error", "Please select a CSV file.")
        return
    
    selected_algo = algo_var.get()
    
    try:
        data = load_data(csv_file)
        transactions = preprocess_data(data)

        if selected_algo == "Apriori":
            frequent_itemsets, rules = run_apriori(transactions, min_support=0.1)
        elif selected_algo == "FP Growth":
            frequent_itemsets, rules = run_fp_growth(transactions, min_support=0.1)
        
        output = format_output(frequent_itemsets, rules)
        display_output(output)
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

root = tk.Tk()
root.title("Market Basket Analysis")

root.geometry("800x400")
root.configure(bg="lightblue")

font_large = ("Arial", 14)
font_small = ("Arial", 12)

tk.Label(root, text="CSV File:", font=font_large, bg="lightblue").grid(row=0, column=0, padx=10, pady=10, sticky="e")
file_entry = tk.Entry(root, width=50, font=font_small)
file_entry.grid(row=0, column=1, padx=10, pady=10)
browse_button = tk.Button(root, text="Browse", command=browse_file, font=font_small)
browse_button.grid(row=0, column=2, padx=10, pady=10)

algo_var = tk.StringVar(value="Apriori")
tk.Label(root, text="Select Algorithm:", font=font_large, bg="lightblue").grid(row=1, column=0, padx=10, pady=10, sticky="e")
apriori_radio = tk.Radiobutton(root, text="Apriori", variable=algo_var, value="Apriori", font=font_small, bg="lightblue")
apriori_radio.grid(row=1, column=1, sticky="w")
fp_growth_radio = tk.Radiobutton(root, text="FP Growth", variable=algo_var, value="FP Growth", font=font_small, bg="lightblue")
fp_growth_radio.grid(row=2, column=1, sticky="w")

output_button = tk.Button(root, text="Get Output", command=run_algorithm, font=font_large)
output_button.grid(row=3, column=1, pady=30)

root.mainloop()