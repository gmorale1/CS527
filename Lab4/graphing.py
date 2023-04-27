import numpy as np
import concurrent.futures
import subprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(11.7,8.27)})
results = pd.read_csv("./results.csv")

#graphing
converging_df = results[results['fitness'] <= 1e-10]
diverging_df = results[results['fitness'] > 1e-10]
# conv_rosenbrock = converging_df[converging_df['func'] == "Rosenbrock"]
# div_rosenbrock = diverging_df[diverging_df['func'] == "Rosenbrock"]
# conv_booth = converging_df[converging_df['func'] == "Booth"]
# div_booth = diverging_df[diverging_df['func'] == "Booth"]

converging_defaults = converging_df[(converging_df["social"] == 1) & (converging_df["cognition"] == 1)]
diverging_defaults = diverging_df[(diverging_df["social"] == 1) & (diverging_df["cognition"] == 1)]
#converging
#num_particles
plt.xticks(np.arange(10,110,step=10))
plot = sns.boxplot(x="num_particles", y="epoch_stop", hue="func", data=converging_defaults)
fig = plot.get_figure()
fig.savefig("./figures/converging_num_particles.png")
plt.clf()

#inertia
# plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plot = sns.boxplot(x="inertia", y="epoch_stop", hue="func", data=converging_defaults)
plt.xticks(range(1, 11)[::2], labels=np.round(np.arange(.1, 1.1, .1), 2)[::2])
fig = plot.get_figure()
fig.savefig("./figures/converging_inertia.png")
plt.clf()

sns.set(rc={'figure.figsize':(11.7,15)})

#cognition
plot = sns.boxplot(x="cognition", y="epoch_stop", hue="func", data=converging_df[(converging_df["social"] >= 0 ) & (converging_df["cognition"] >= 0)])
# plt.xticks(range(1, 41)[::2], labels=np.round(np.arange(.1, 4.1, .1), 2)[::2])
plt.xticks(range(1, 41)[::2], labels=np.round(np.arange(.1, 4.1, .1), 2)[::2])
# plt.xticks(np.arange(0.1,4.5,step=0.5))
# plot.set_xticks(np.arange(0.1,4.0,step=0.5))
fig = plot.get_figure()
fig.savefig("./figures/converging_cognition.png")
plt.clf()

#social
# plt.xticks(np.arange(0.1,4.0,0.1))
plot = sns.boxplot(x="social", y="epoch_stop", hue="func", data=converging_df)
plt.xticks(range(1, 41)[::2], labels=np.round(np.arange(.1, 4.1, .1), 2)[::2])
fig = plot.get_figure()
fig.savefig("./figures/converging_social.png")
plt.clf()

sns.set(rc={'figure.figsize':(11.7,8.27)})

#diverging
#num_particles
# plt.xticks([10,20,30,40,50,60,70,80,90,100])
plot = sns.boxplot(x="num_particles", y="epoch_stop", hue="func", data=diverging_defaults)
plt.xticks(range(1, 10)[::2], labels=np.round(np.arange(10, 110, 10), 2)[::2])
fig = plot.get_figure()
fig.savefig("./figures/diverging_num_particles.png")
plt.clf()

#inertia
# plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plot = sns.boxplot(x="inertia", y="epoch_stop", hue="func", data=diverging_defaults)
plt.xticks(range(1, 11)[::2], labels=np.round(np.arange(.1, 1.1, .1), 2)[::2])
fig = plot.get_figure()
fig.savefig("./figures/diverging_inertia.png")
plt.clf()

sns.set(rc={'figure.figsize':(11.7,15)})

#cognition
# plt.xticks(np.arange(0.1,4.0,0.1))
plot = sns.boxplot(x="cognition", y="epoch_stop", hue="func", data=diverging_df)
plt.xticks(range(1, 41)[::2], labels=np.round(np.arange(.1, 4.1, .1), 2)[::2])
fig = plot.get_figure()
fig.savefig("./figures/diverging_cognition.png")
plt.clf()

#social
# plt.xticks(np.arange(0.1,4.0,0.1))
plot = sns.boxplot(x="social", y="epoch_stop", hue="func", data=diverging_df)
plt.xticks(range(1, 41)[::2], labels=np.round(np.arange(.1, 4.1, .1), 2)[::2])
fig = plot.get_figure()
fig.savefig("./figures/diverging_social.png")
plt.clf()

sns.set(rc={'figure.figsize':(11.7,8.27)})

#cognition and social heatmap
#converging
# plt.xticks(np.arange(0.1,4.0,0.1))
converging_heatmap = converging_df[(converging_df["num_particles"] == 40) & (converging_df["inertia"] == 0.5) & (converging_df["func"] == "Rosenbrock")]
heatmap_data = converging_heatmap.pivot_table(index='cognition', columns='social', values='epoch_stop', aggfunc='mean')
plot = sns.heatmap(heatmap_data, cmap='YlGnBu', fmt="1g")
plt.xticks(range(1, 41)[::2], labels=np.round(np.arange(.1, 4.1, .1), 2)[::2])
fig = plot.get_figure()
fig.savefig("./figures/converging_heatmap_rosen.png")
plt.clf()

# plt.xticks(np.arange(0.1,4.0,0.1))
converging_heatmap = converging_df[(converging_df["num_particles"] == 40) & (converging_df["inertia"] == 0.5) & (converging_df["func"] == "Booth")]
heatmap_data = converging_heatmap.pivot_table(index='cognition', columns='social', values='epoch_stop', aggfunc='mean')
plot = sns.heatmap(heatmap_data, cmap='YlGnBu', fmt="1g")
plt.xticks(range(1, 41)[::2], labels=np.round(np.arange(.1, 4.1, .1), 2)[::2])
fig = plot.get_figure()
fig.savefig("./figures/converging_heatmap_booth.png")
plt.clf()

#diverging
# plt.xticks(np.arange(0.1,4.0,0.1))
diverging_heatmap = diverging_df[(diverging_df["num_particles"] == 40) & (diverging_df["inertia"] == 0.5) & (diverging_df["func"] == "Rosenbrock")]
heatmap_data = diverging_heatmap.pivot_table(index='cognition', columns='social', values='epoch_stop', aggfunc='count')
plot = sns.heatmap(heatmap_data, cmap='YlGnBu', fmt="1g")
plt.xticks(range(1, 41)[::2], labels=np.round(np.arange(.1, 4.1, .1), 2)[::2])
fig = plot.get_figure()
fig.savefig("./figures/diverging_heatmap_rosen.png")
plt.clf()

# plt.xticks(np.arange(0.1,4.0,0.1))
diverging_heatmap = diverging_df[(diverging_df["num_particles"] == 40) & (diverging_df["inertia"] == 0.5) & (diverging_df["func"] == "Booth")]
heatmap_data = diverging_heatmap.pivot_table(index='cognition', columns='social', values='epoch_stop', aggfunc='count')
plot = sns.heatmap(heatmap_data, cmap='YlGnBu', fmt="1g")
plt.xticks(range(1, 41)[::2], labels=np.round(np.arange(.1, 4.1, .1), 2)[::2])
fig = plot.get_figure()
fig.savefig("./figures/diverging_heatmap_booth.png")
plt.clf()
