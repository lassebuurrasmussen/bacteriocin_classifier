import pandas as pd
from tqdm import tqdm
import joblib
import time

# %% Scrape camp
tables = []
na_counter = 0
for camp_id in tqdm(range(8400)):
    url = f"http://www.camp3.bicnirrh.res.in/seqDisp.php?id=CAMPSQ{camp_id}"
    table = pd.read_html(url)[2]
    tables.append(table)

    if table[1].isna().all():
        na_counter += 1
        print("\nNAN counter: ", na_counter)

# %%
# joblib.dump(tables, "camp_temp_result")  # Save temporary scrape result
# tables = joblib.load("camp_temp_result")  # Load temporary scrape result

# %% Convert to Pandas data frame
tables_fixed = []
for table in tqdm(tables):
    t = table.copy().set_index(0)
    if t.isna().all(axis=None):
        continue

    i = t.index.tolist()
    assert i[0][:6] == 'CAMPSQ'
    i[0] = 'CAMPSQ'
    t.index = i
    tables_fixed.append(t.T)

df: pd.DataFrame = pd.concat(tables_fixed, sort=False)

# %% Clean and save file
df.reset_index(drop=True, inplace=True)
assert not df.isna().all(axis=1).any()
df.drop_duplicates(inplace=True)

df.rename(columns={'CAMPSQ': 'CAMP_ID'}, inplace=True)
df.columns = df.columns.str.replace(' :', '')
df.columns = df.columns.str.replace(':', '')
assert not df['CAMP_ID'].duplicated().any()

# df.to_csv(f"data/camp/camp_database{time.strftime('%m-%d-%y')}.csv", index=False)

# %%
