import pandas as pd

data = pd.read_excel(r'../data/UC_all_divided_data.xlsx')

total_count = dict()
for i in data.loc[:, "image_source"].unique():
    total_count[i] = data[data["image_source"]==i].shape[0]
print(total_count)

for i in total_count:
    class_count = dict()
    for j in range(4):
        class_count[j] = data[(data["image_source"]==i)&(data["label"]==j)].shape[0]
    print(f"{i}的类别分布为{class_count}")
    print("="*50)
