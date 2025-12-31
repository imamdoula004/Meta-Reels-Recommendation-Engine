# main.py - Reels + Analytics + Behavioral Integration (robust)
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import random
import time

# ------------------------
# Step 1: Load datasets via file explorer
# ------------------------
Tk().withdraw()
file_paths = askopenfilenames(title="Select CSV datasets (Reels + Analytics + Behavioral)", filetypes=[("CSV files", "*.csv")])
if not file_paths or len(file_paths) < 2:
    print("Please select at least Reels and Behavioral datasets. Exiting.")
    exit()

# Identify datasets
reels_df = None
analytics_df = None
behavior_df = None

for path in file_paths:
    fname = path.lower()
    if "reel" in fname:
        reels_df = pd.read_csv(path)
    elif "analytics" in fname:
        analytics_df = pd.read_csv(path)
    elif "behavior" in fname:
        behavior_df = pd.read_csv(path)

if reels_df is None or behavior_df is None:
    print("Could not find Reels or Behavioral datasets. Exiting.")
    exit()

# ------------------------
# Step 2: Merge Reels + Analytics (robust)
# ------------------------
analytics_available = False
if analytics_df is not None:
    # Attempt to find a common column
    common_cols = set(reels_df.columns).intersection(set(analytics_df.columns))
    if len(common_cols) > 0:
        merge_col = list(common_cols)[0]
        merged_reels = reels_df.merge(analytics_df, how='left', on=merge_col, suffixes=('_reel','_analytics'))
        analytics_available = True
    else:
        print("Warning: No common column between Reels and Analytics. Using synthetic analytics features.")
        merged_reels = reels_df.copy()
else:
    merged_reels = reels_df.copy()

# Ensure meta_link column exists
if 'link_post' not in merged_reels.columns:
    # Try to use any unique ID column
    id_col = merged_reels.columns[0]
    merged_reels['link_post'] = merged_reels[id_col].apply(lambda x: f"https://www.instagram.com/reel/{x}/")

# ------------------------
# Step 3: Merge Behavioral dataset
# ------------------------
if 'user_id' in behavior_df.columns:
    merged_reels['user_id'] = merged_reels.get('user_id', 0)
    merged_df = merged_reels.merge(behavior_df, how='left', on='user_id', suffixes=('','_behavior'))
else:
    print("Behavioral dataset does not contain 'user_id'. Using Reels only.")
    merged_df = merged_reels.copy()

# ------------------------
# Step 4: Preprocess dataset
# ------------------------
def preprocess_dataset(df):
    df = df.copy()
    n = len(df)
    df['clicked'] = np.random.choice([0,1], n)  # target variable

    # Numeric features
    numeric_features = ['age','recent_interactions_meta','historic_interactions_meta','present_interactions_external',
                        'likes','comments','shares']
    for col in numeric_features:
        if col not in df.columns:
            df[col] = np.random.randint(0,50,n)

    # Categorical features
    categorical_defaults = {
        'gender': ['male','female'],
        'psyche_type': ['introvert','extrovert','ambivert'],
        'mood': ['happy','sad','neutral'],
        'fashion_sense': ['casual','formal','sporty'],
        'lifestyle': ['luxury','average','simple'],
        'living_standards': ['high','medium','low'],
        'location': ['home','work','outside'],
        'present_setting': ['home','office','outdoor'],
        'time_of_day': ['morning','afternoon','evening','night']
    }
    for col, choices in categorical_defaults.items():
        if col not in df.columns:
            df[col] = np.random.choice(choices, n)
        else:
            df[col] = df[col].fillna(random.choice(choices))

    df['recent_action'] = np.random.choice(df['link_post'], n)
    df['recent_interactions_meta'] = df.get('recent_interactions_meta', np.random.randint(0,5,n))
    df['historic_interactions_meta'] = df.get('historic_interactions_meta', np.random.randint(5,50,n))
    df['present_interactions_external'] = df.get('present_interactions_external', np.random.randint(0,5,n))

    return df

user_df = preprocess_dataset(merged_df.sample(n=min(100, len(merged_df)), random_state=42))

# ------------------------
# Step 5: Features & model
# ------------------------
numeric_features = ['age','recent_interactions_meta','historic_interactions_meta','present_interactions_external',
                    'likes','comments','shares']
categorical_features = ['gender','psyche_type','mood','fashion_sense','lifestyle','living_standards',
                        'location','present_setting','time_of_day']

user_df[numeric_features] = user_df[numeric_features].fillna(0)
user_df[categorical_features] = user_df[categorical_features].fillna('unknown')

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(user_df[categorical_features])
X_num = user_df[numeric_features].values
X = np.hstack([X_num, X_cat])
y = user_df['clicked'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# ------------------------
# Step 6: Recommendation function
# ------------------------
def recommend_dynamic(user_df, top_n=5):
    user_X_cat = encoder.transform(user_df[categorical_features])
    user_X_num = user_df[numeric_features].values
    user_X = np.hstack([user_X_num, user_X_cat])

    probs = model.predict_proba(user_X)[:,1]
    user_df['score'] = probs
    top_recs = user_df.sort_values('score', ascending=False).head(top_n).copy()
    top_recs['meta_link'] = top_recs['link_post']

    explanations = []
    for _, row in top_recs.iterrows():
        reasons = [
            f"Mood: {row['mood']}",
            f"Recent action: {row['recent_action']}",
            f"Recent interactions (Meta): {row['recent_interactions_meta']}",
            f"Historic interactions (Meta): {row['historic_interactions_meta']}",
            f"Time of day: {row['time_of_day']}",
            f"Location: {row['location']}, Setting: {row['present_setting']}",
            f"Likes: {row.get('likes','N/A')}, Comments: {row.get('comments','N/A')}, Shares: {row.get('shares','N/A')}",
            f"Engagement probability: {row['score']:.2f}"
        ]
        explanations.append("; ".join(reasons))
    top_recs['reason'] = explanations
    return top_recs[['meta_link','score','reason']]

# ------------------------
# Step 7: Dynamic session with feedback
# ------------------------
print(f"\n--- Simulating dynamic session with feedback for user ---\n")
for step in range(3):
    user_df['mood'] = np.random.choice(['happy','sad','neutral'], len(user_df))
    user_df['present_setting'] = np.random.choice(['home','office','outdoor'], len(user_df))
    user_df['recent_action'] = np.random.choice(user_df['link_post'], len(user_df))
    user_df['time_of_day'] = np.random.choice(['morning','afternoon','evening','night'], len(user_df))

    recs = recommend_dynamic(user_df, top_n=5)

    clicked_posts = []
    for idx, row in recs.iterrows():
        if random.random() < row['score']:
            clicked_posts.append(row['meta_link'])

    print(f"--- Session Step {step+1} ---")
    for i, row in recs.iterrows():
        click_status = "Clicked" if row['meta_link'] in clicked_posts else "Not Clicked"
        print(f"Post Link: {row['meta_link']} [{click_status}]")
        print(f"Score: {row['score']:.2f}")
        print(f"Why suggested: {row['reason']}\n")

    time.sleep(3)
