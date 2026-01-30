import pandas as pd

# Read the CSV file
df = pd.read_csv('d:/2026-repo/2026_MCM_Problem_C_Data.csv')

# Initialize a list to store the results
results = []

# Ensure we cover seasons 1 to 34
for season in range(1, 35):
    # Filter the dataframe for the current season
    season_df = df[df['season'] == season]
    
    # Count the number of participants (rows) in this season
    participant_count = len(season_df)
    
    results.append({'Season': season, 'Participant Count': participant_count})

# Create a DataFrame for the output
output_df = pd.DataFrame(results)

# Print the table in a readable format
print(output_df.to_string(index=False))

# Save to a csv file for reference
output_df.to_csv('d:/2026-repo/participant_counts_per_season.csv', index=False)
