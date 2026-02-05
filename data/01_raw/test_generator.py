import pandas as pd
import matplotlib.pyplot as plt

# Loading datas
datas = pd.read_csv('examens_tonaux.csv')

# Create x axis
init_freq = 125
abscisses = [str(init_freq * (2 ** i)) for i in range(7)]

# Splitting before and after exam
datas_before = datas[[col for col in datas.columns if 'before' in col]]
datas_after = datas[[col for col in datas.columns if 'after' in col]]

# Plotting
fig = plt.figure()
step = 500
for i in range(0, len(datas_before)-step + 10, step):
    plt.plot(abscisses, datas_before.loc[i:i+step].describe().loc['mean'].values)
    plt.plot(abscisses, datas_after.loc[i:i+step].describe().loc['mean'].values)

# Reverse the y-axis
plt.gca().invert_yaxis()

# Get the current axes object
ax = plt.gca()

# Move the x-axis above the graph
ax.spines['bottom'].set_position('zero')
ax.spines['bottom'].set_zorder(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.ylabel('Perte tonale')
plt.legend()
plt.show()