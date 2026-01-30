import pandas as pd
import numpy as np
import re

# Load the data
input_file = 'd:/2026-repo/2026_MCM_Problem_C_Data.csv'
df = pd.read_csv(input_file)

# Step A: Calculate Weekly Total Judge Score (J_{i,t})
# Create columns for total scores if they don't exist (conceptually, we will use them for logic)
# We will iterate and compute on the fly or add columns. Adding columns is good for inspection.

# Helper function to get sum of scores for a week
def get_weekly_score(row, week_num):
    total_score = 0
    has_score = False
    for judge in range(1, 5):
        col_name = f'week{week_num}_judge{judge}_score'
        if col_name in row.index:
            val = row[col_name]
            # Handle N/A and conversions
            try:
                score = pd.to_numeric(val, errors='coerce')
                if not np.isnan(score):
                    total_score += score
                    has_score = True
            except:
                pass
    return total_score if has_score else 0

# Step D: Infer Withdrew week and update 'results' column
# Logic: Find the last week with J_{i,t} > 0. Let this be last_active_week.
# Update 'results' to "Withdrew (Eliminated Week {last_active_week})"

# We apply this logically.
# 1. Identify rows where 'results' contains "Withdrew" (case-insensitive probably safter, but data seems consistent)
# 2. For these rows, calculate J_{i,t} for t=1..11
# 3. Find the max t where J_{i,t} > 0.
# 4. Construct the new string.

# Let's iterate through the dataframe and process "Withdrew" cases specifically first as requested.
# The user wants to modify the 'results' column in the original data style.

def process_withdrawals(row):
    result_str = str(row['results'])
    if 'Withdrew' in result_str or 'Quit' in result_str: # Handling "Quit" just in case, though user said Withdrew
        
        last_active_week = 0
        for week in range(1, 12):
            score = get_weekly_score(row, week)
            if score > 0:
                last_active_week = week
        
        # If we found an active week, append the info
        # User format: "Withdrew (Eliminated Week j)"
        # Note: If they withdrew, they likely performed that week and then left, or left before next week.
        # The user says: "认为他是在 last_active_week 那周之后退出... (Eliminated Week j) j就是退出的时间"
        # So conceptually they are eliminated "at week j".
        
        if last_active_week > 0:
            # Check if it already has parenthetical info to avoid double adding if run multiple times
            if '(Eliminated Week' not in result_str:
                return f"{result_str} (Eliminated Week {last_active_week})"
    
    return row['results']

# Apply the processing
df['results'] = df.apply(process_withdrawals, axis=1)

# Step B & C are logic guides for modeling ("Step B: 判定 active", "Step C: 判定被投票淘汰"), 
# but the specific request for modification seems to be concentrated on Step D ("按这个规则 对几个出现withdrew的处理...").
# However, to be helpful for "Data Cleaning", I should probably output the cleaned dataset 
# that might include these calculated fields or just the corrected text column.
# The user asked for "最推荐的数据整理流程", implying they want the data ready for modeling.
# But the explicit instruction for modification is "按这个规则 对几个出现withdrew的处理".

# I will save the modified CSV.

output_file = 'd:/2026-repo/2026_MCM_Problem_C_Data_Cleaned.csv'
df.to_csv(output_file, index=False)

print(f"Processed {len(df)} rows.")
print("Updated 'results' column for Withdrew cases.")

# Verify a few examples
withdrew_rows = df[df['results'].str.contains('Withdrew', na=False)]
print(withdrew_rows[['celebrity_name', 'season', 'results']].to_string())
