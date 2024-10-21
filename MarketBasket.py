import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tkinter import ttk

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

def load_data(file_path):
    return pd.read_csv(file_path)

def run_apriori(transactions, min_support=0.01):
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.15)
    return frequent_itemsets, rules

def run_fp_growth(transactions, min_support=0.01):
    frequent_itemsets = fpgrowth(transactions, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.15)
    return frequent_itemsets, rules

def preprocess_data(data):
    transactions = data.groupby(['Transaction', 'Item'])['Item'].count().unstack().fillna(0)
    transactions = transactions.astype(bool)
    
    return transactions

def format_output(frequent_itemsets, rules):
    output = "Table 1: Frequent Itemsets with Support\n"
    output += "{:<40} {:<10}\n".format("Itemset", "Support")
    output += "-" * 50 + "\n"
    
    for _, row in frequent_itemsets.iterrows():
        itemset = ", ".join([str(item) if item is not None else 'None' for item in row['itemsets']])
        output += "{:<40} {:<10.2%}\n".format(itemset, row['support'])
    
    output += "\nTable 2: Association Rules\n"
    output += "{:<40} {:<40} {:<10} {:<10}\n".format("Antecedent", "Consequent", "Support", "Confidence")
    output += "-" * 100 + "\n"
    
    for _, row in rules.iterrows():
        antecedents = ', '.join([str(item) if item is not None else 'None' for item in list(row['antecedents'])])
        consequents = ', '.join([str(item) if item is not None else 'None' for item in list(row['consequents'])])
        output += "{:<40} {:<40} {:<10.2%} {:<10.2%}\n".format(antecedents, consequents, row['support'], row['confidence'])
    
    return output

def display_output(output):
    output_window = tk.Toplevel(root)
    output_window.title("Analysis Output")
    
    text_widget = tk.Text(output_window, wrap="none", width=100, height=30, bg="#282828", fg="#f0f0f0", font=("Helvetica", 12))
    text_widget.pack(side="left", fill="both", expand=True)
    
    scrollbar = tk.Scrollbar(output_window, command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")
    
    text_widget.config(yscrollcommand=scrollbar.set)
    text_widget.insert(tk.END, output)
    text_widget.config(state=tk.DISABLED)

def display_csv_data():
    csv_file = file_entry.get()
    if not csv_file:
        messagebox.showwarning("Input Error", "Please select a CSV file.")
        return
    
    data = load_data(csv_file)
    csv_output_window = tk.Toplevel(root)
    csv_output_window.title("CSV Data")
    
    text_widget = tk.Text(csv_output_window, wrap="none", width=100, height=30, bg="#282828", fg="#f0f0f0", font=("Helvetica", 12))
    text_widget.pack(side="left", fill="both", expand=True)
    
    scrollbar = tk.Scrollbar(csv_output_window, command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")
    
    text_widget.config(yscrollcommand=scrollbar.set)
    text_widget.insert(tk.END, data.to_string(index=False))
    text_widget.config(state=tk.DISABLED)

def visualize_data():
    csv_file = file_entry.get()
    if not csv_file:
        messagebox.showwarning("Input Error", "Please select a CSV file.")
        return

    data = load_data(csv_file)
    transactions = preprocess_data(data)
    
    frequent_itemsets, rules = run_apriori(transactions, min_support=0.01)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='support', y='itemsets', data=frequent_itemsets.nlargest(10, 'support'), palette='viridis')
    plt.title('Top 10 Frequent Itemsets')
    plt.xlabel('Support')
    plt.ylabel('Itemsets')
    plt.show()
    
    if not rules.empty:
        heatmap_data = rules.pivot_table(index='antecedents', columns='consequents', values='confidence', aggfunc='mean')

        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', cbar=True, fmt='.2f')
        plt.title('Heatmap of Rule Confidence', fontsize=16)
        plt.xlabel('Consequents', fontsize=12)
        plt.ylabel('Antecedents', fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        messagebox.showinfo("No Rules", "No association rules found for the given data.")

apriori_time = None
fp_growth_time = None

def show_efficiency():
    efficiency_window = tk.Toplevel(root)
    efficiency_window.title("Algorithm Efficiency")
    
    text_widget = tk.Text(efficiency_window, wrap="none", width=50, height=10, bg="#282828", fg="#f0f0f0", font=("Helvetica", 12))
    text_widget.pack(side="left", fill="both", expand=True)
    
    scrollbar = tk.Scrollbar(efficiency_window, command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")
    
    text_widget.config(yscrollcommand=scrollbar.set)

    if apriori_time is not None:
        text_widget.insert(tk.END, f"Apriori Execution Time: {apriori_time:.4f} seconds\n")
    else:
        text_widget.insert(tk.END, "Apriori Execution Time: None\n")
    
    if fp_growth_time is not None:
        text_widget.insert(tk.END, f"FP Growth Execution Time: {fp_growth_time:.4f} seconds\n")
    else:
        text_widget.insert(tk.END, "FP Growth Execution Time: None\n")
    
    if apriori_time is not None and fp_growth_time is not None:
        if apriori_time < fp_growth_time:
            text_widget.insert(tk.END, "Apriori is faster than FP Growth.\n")
        else:
            text_widget.insert(tk.END, "FP Growth is faster than Apriori.\n")
    
    text_widget.config(state=tk.DISABLED)

def run_algorithm():
    global apriori_time, fp_growth_time
    
    csv_file = file_entry.get()
    if not csv_file:
        messagebox.showwarning("Input Error", "Please select a CSV file.")
        return
    
    selected_algo = algo_var.get()
    
    try:
        data = load_data(csv_file)
        transactions = preprocess_data(data)
        
        start_time = time.time()
        frequent_itemsets, rules = run_apriori(transactions, min_support=0.03)
        apriori_time = time.time() - start_time
        start_time = time.time()
        frequent_itemsets, rules = run_fp_growth(transactions, min_support=0.03)
        fp_growth_time = time.time() - start_time

        start_time = time.time()
        if selected_algo == "Apriori":
            frequent_itemsets, rules = run_apriori(transactions, min_support=0.03)
            apriori_time = time.time() - start_time
        elif selected_algo == "FP Growth":
            frequent_itemsets, rules = run_fp_growth(transactions, min_support=0.03)
            fp_growth_time = time.time() - start_time

        output = format_output(frequent_itemsets, rules)
        display_output(output)
        
        show_efficiency()
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

root = tk.Tk()
root.title("Market Basket Analysis")
root.geometry("700x400")
root.configure(bg="#121212")
root.resizable(False, False)

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", padding=10, borderwidth=0, background="#202020", foreground="#e8e8e8", font=("Helvetica", 12))
style.map("TButton", background=[("active", "#6200EA")])
style.configure("TRadiobutton", background="#121212", foreground="#f0f0f0", font=("Helvetica", 12))

header_frame = tk.Frame(root, bg="#121212", padx=20, pady=10)
header_frame.pack(fill=tk.X)

tk.Label(header_frame, text="Market Basket Analysis", font=("Helvetica", 18, "bold"), bg="#121212", fg="white").pack()

input_frame = tk.Frame(root, bg="#121212", padx=20, pady=20)
input_frame.pack(pady=10)

tk.Label(input_frame, text="CSV File:", font=("Helvetica", 12), bg="#121212", fg="#f0f0f0").grid(row=0, column=0, sticky="w")
file_entry = ttk.Entry(input_frame, width=40, font=("Helvetica", 12))
file_entry.grid(row=0, column=1, padx=10)
browse_button = ttk.Button(input_frame, text="Browse", command=browse_file)
browse_button.grid(row=0, column=2)

algo_frame = tk.Frame(root, bg="#121212", padx=20, pady=10)
algo_frame.pack(pady=10)

tk.Label(algo_frame, text="Select Algorithm:", font=("Helvetica", 12), bg="#121212", fg="#f0f0f0").pack(anchor="w")
algo_var = tk.StringVar(value="Apriori")

apriori_radio = ttk.Radiobutton(algo_frame, text="Apriori", variable=algo_var, value="Apriori", style="TRadiobutton")
apriori_radio.pack(anchor="w")

fp_growth_radio = ttk.Radiobutton(algo_frame, text="FP Growth", variable=algo_var, value="FP Growth", style="TRadiobutton")
fp_growth_radio.pack(anchor="w")

button_frame = tk.Frame(root, bg="#121212", padx=20, pady=10)
button_frame.pack(pady=10)

visualize_button = ttk.Button(button_frame, text="Visualize Data", command=visualize_data)
visualize_button.grid(row=0, column=2, padx=10)

display_button = ttk.Button(button_frame, text="Display CSV Data", command=display_csv_data)
display_button.grid(row=0, column=0, padx=10)

analyze_button = ttk.Button(button_frame, text="Run Algorithm", command=run_algorithm)
analyze_button.grid(row=0, column=1, padx=10)

root.mainloop()
