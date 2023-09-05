##################### Key/Danceability İlişkisi ############################

df['dance_prof'] = df['danceability'].apply(lambda x: 1 if x > 0.5 else 0)
grouped = df.groupby(['dance_prof', 'key']).size().unstack(fill_value=0)
"""
grouped
key           0     1     2     3     4   ...    7     8     9     10    11
dance_prof                                ...                              
0           4412  3364  4391  1303  3159  ...  4401  2240  3971  2276  2627
1           8649  7408  7253  2267  5849  ...  8843  5120  7342  5180  6655
[2 rows x 12 columns]
pd.set_option("display.max_columns", None)
grouped
key           0     1     2     3     4     5     6     7     8     9     10  \
dance_prof                                                                     
0           4412  3364  4391  1303  3159  3085  2348  4401  2240  3971  2276   
1           8649  7408  7253  2267  5849  6283  5573  8843  5120  7342  5180   
key           11  
dance_prof        
0           2627  
1           6655 
"""

df['dance_prof'] = df['danceability'].apply(lambda x: 1 if x > 0.7 else 0)
grouped = df.groupby(['dance_prof', 'key']).size().unstack(fill_value=0)

"""
key            0     1     2     3     4     5     6      7     8     9   \
dance_prof                                                                 
0           10121  7715  9416  2820  7077  7264  5794  10069  5384  8928   
1            2940  3057  2228   750  1931  2104  2127   3175  1976  2385   
key           10    11  
dance_prof              
0           5446  6719  
1           2010  2563  
"""
