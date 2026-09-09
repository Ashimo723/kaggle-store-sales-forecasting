import pandas as pd, numpy as np
D="/Users/hiromuashida/Desktop/Codex/kaggle/store-sales/data/"
tr=pd.read_csv(D+"train.csv",parse_dates=["date"])
te=pd.read_csv(D+"test.csv",parse_dates=["date"])
st=pd.read_csv(D+"stores.csv"); ho=pd.read_csv(D+"holidays_events.csv",parse_dates=["date"])
oil=pd.read_csv(D+"oil.csv",parse_dates=["date"]); tx=pd.read_csv(D+"transactions.csv",parse_dates=["date"])

print("== 期間 ==")
print("train:",tr.date.min().date(),"->",tr.date.max().date(),"日数",tr.date.nunique())
print("test :",te.date.min().date(),"->",te.date.max().date(),"日数",te.date.nunique())
print("store_nbr:",tr.store_nbr.nunique(),"family:",tr.family.nunique(),"系列数",tr.store_nbr.nunique()*tr.family.nunique())
print("train行/日 =",len(tr)/tr.date.nunique(),"  test行:",len(te))
print("欠測日(train全日付が連続か):", pd.date_range(tr.date.min(),tr.date.max()).difference(tr.date.unique()))

print("\n== sales 分布 ==")
print(tr.sales.describe())
print("sales==0 比率: %.3f"%(tr.sales==0).mean())
print("log1p後 std: %.3f"%np.log1p(tr.sales).std())

print("\n== 店舗別 開店日(最初に非ゼロ売上が出た日) ==")
first=tr[tr.sales>0].groupby("store_nbr").date.min()
print(first[first>tr.date.min()].to_string())

print("\n== family別 ゼロ比率 上位/下位 ==")
z=tr.groupby("family").sales.apply(lambda s:(s==0).mean()).sort_values()
print(z.head(5).to_string()); print("...");print(z.tail(8).to_string())

print("\n== 直近: 各系列で最後の N 日すべてゼロの系列数 ==")
last=tr[tr.date>=tr.date.max()-pd.Timedelta(days=59)]
g=last.groupby(["store_nbr","family"]).sales.sum()
print("直近60日 合計0 の系列数:",(g==0).sum(),"/",len(g))

print("\n== onpromotion ==")
print("train mean %.3f max %d / test mean %.3f max %d"%(tr.onpromotion.mean(),tr.onpromotion.max(),te.onpromotion.mean(),te.onpromotion.max()))
print("train 2014年以前のonpromotion合計:",tr[tr.date<"2015-01-01"].onpromotion.sum())

print("\n== transactions ==")
print("期間:",tx.date.min().date(),"->",tx.date.max().date(),"  ※test期間に無い→未来特徴として使えない")

print("\n== oil ==")
print("期間:",oil.date.min().date(),"->",oil.date.max().date(),"欠損:",oil.dcoilwtico.isna().sum(),"/",len(oil))

print("\n== holidays type/locale ==")
print(ho.type.value_counts().to_string()); print(ho.locale.value_counts().to_string())
print("transferred=True:",ho.transferred.sum())
print("test期間に重なる祝日:"); print(ho[(ho.date>=te.date.min())&(ho.date<=te.date.max())].to_string())

print("\n== stores ==")
print(st.type.value_counts().to_string()); print("cluster数:",st.cluster.nunique(),"city数:",st.city.nunique(),"state数:",st.state.nunique())

print("\n== 全店合計売上の月次(直近12ヶ月) ==")
m=tr.groupby(tr.date.dt.to_period("M")).sales.sum()
print(m.tail(12).to_string())
print("\n== 1/1 の売上 ==")
print(tr[(tr.date.dt.month==1)&(tr.date.dt.day==1)].groupby("date").sales.sum().to_string())
print("\n== 曜日別平均 ==")
print(tr.groupby(tr.date.dt.dayofweek).sales.mean().to_string())
print("\n== 日付(給料日)別平均 ==")
print(tr.groupby(tr.date.dt.day).sales.mean().round(2).to_string())
