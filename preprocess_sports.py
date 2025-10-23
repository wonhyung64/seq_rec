#%%
import pandas as pd
from tqdm import tqdm


#%%
file_dir = "/home/optim1/Desktop/won/tendency_bias/data/Sports_6"


#%%
df_sport = pd.read_parquet(f"{file_dir}/lare_Sports_and_Outdoors.parquet")
filtering = 0

while True:

    cond = 0

    item_interaction_num = df_sport["asin"].value_counts()
    if len(item_interaction_num[item_interaction_num<5]) != 0:
        cond +=1
        valid_item = (item_interaction_num[item_interaction_num>=5]).index.tolist()
        df_sport = df_sport[df_sport["asin"].isin(valid_item)]

    user_interaction_num = df_sport["user_id"].value_counts()
    if len(user_interaction_num[user_interaction_num<5]) != 0:
        cond +=1
        valid_user = (user_interaction_num[user_interaction_num>=5]).index.tolist()
        df_sport = df_sport[df_sport["user_id"].isin(valid_user)]
    
    if cond == 0:
        break
    else:
        filtering += 1
        print(f"Filtering {filtering} -> {len(df_sport)}")

df_sport.to_parquet(f"{file_dir}/lare_Sports_and_Outdoors_valid.parquet")


#%%
df_filtered2 = pd.read_parquet(f"{file_dir}/lare_Sports_and_Outdoors_valid.parquet")
user_list = df_filtered2.user_id.unique().tolist()
df_list = []
for u in tqdm(user_list):
    # 1) 타임스탬프를 datetime64[ns] 로 변환
    df = df_filtered2[df_filtered2.user_id==u].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2) user-별, 시각순으로 정렬
    df.sort_values(["user_id", "timestamp"], inplace=True)

    # 3) 직전 이벤트와의 시간 차이를 계산
    df["delta"] = df.groupby("user_id")["timestamp"].diff()

    # 4) ① 첫 행(NaT) 혹은 ② 30 분 초과 ⇒ 새 세션 플래그
    df["new_session"] = (df["delta"].isna()) | (df["delta"] > pd.Timedelta(minutes=30))

    # 5) 누적합(cumsum)으로 user-별 세션 번호 부여
    df["session_no"] = df.groupby("user_id")["new_session"].cumsum()

    # ────────────────────────────────
    # 선택) 전역 고유 세션 ID 만들기
    df["session_id"] = df["user_id"] + "_" + df["session_no"].astype(str)

    df["session_len"] = df.groupby("session_id")["session_id"].transform("size")

    df_list.append(df)


session_df = pd.concat(df_list).reset_index(drop=True)
session_df.to_parquet(f"{file_dir}/lare_Sports_and_Outdoors_valid_session.parquet")


# %%