from matplotlib import pyplot as pl
import numpy as np
pl.clf()

data = np.genfromtxt("res/experiments_data/synthetic.csv", delimiter=",",skip_header=True)
x = np.array([50,100,200])
spc = data[:, 1]
spc_y = np.ones(3)
spc_y[0] = np.average(spc[:50])
spc_y[1] = np.average(spc[50:100])
spc_y[2] = np.average(spc[100:])

spc_y_lower = np.ones(3)
spc_y_lower[0] = np.quantile(spc[:50], 0.05)
spc_y_lower[1] = np.quantile(spc[50:100], 0.05)
spc_y_lower[2] = np.quantile(spc[100:], 0.05)

spc_y_upper = np.ones(3)
spc_y_upper[0] = np.quantile(spc[:50], 0.95)
spc_y_upper[1] = np.quantile(spc[50:100], 0.95)
spc_y_upper[2] = np.quantile(spc[100:], 0.95)


spc_hull = data[:,2]
spc_hull_y = np.ones(3)

spc_hull_y[0] = np.average(spc_hull[:50])
spc_hull_y[1] = np.average(spc_hull[50:100])
spc_hull_y[2] = np.average(spc_hull[100:])

spc_hull_y_lower = np.ones(3)
spc_hull_y_lower[0] = np.quantile(spc_hull[:50], 0.05)
spc_hull_y_lower[1] = np.quantile(spc_hull[50:100], 0.05)
spc_hull_y_lower[2] = np.quantile(spc_hull[100:], 0.05)

spc_hull_y_upper = np.ones(3)
spc_hull_y_upper[0] = np.quantile(spc_hull[:50], 0.95)
spc_hull_y_upper[1] = np.quantile(spc_hull[50:100], 0.95)
spc_hull_y_upper[2] = np.quantile(spc_hull[100:], 0.95)

spc_pp =  data[:, 3]
spc_pp_y = np.ones(3)
spc_pp_y[0] = np.average(spc_pp[:50])
spc_pp_y[1] = np.average(spc_pp[50:100])
spc_pp_y[2] = np.average(spc_pp[100:])

spc_pp_y_lower = np.ones(3)
spc_pp_y_lower[0] = np.quantile(spc_pp[:50], 0.05)
spc_pp_y_lower[1] = np.quantile(spc_pp[50:100], 0.05)
spc_pp_y_lower[2] = np.quantile(spc_pp[100:], 0.05)

spc_pp_y_upper = np.ones(3)
spc_pp_y_upper[0] = np.quantile(spc_pp[:50], 0.95)
spc_pp_y_upper[1] = np.quantile(spc_pp[50:100], 0.95)
spc_pp_y_upper[2] = np.quantile(spc_pp[100:], 0.95)

pl.plot(x, spc_y, 'k', color='#CC4F1B', marker='o', label=r"SPC")
pl.fill_between(x, spc_y_lower, spc_y_upper,
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

pl.plot(x, spc_hull_y, 'k', color='#1B2ACC', marker='o', label=r"SPC+hull")
pl.fill_between(x, spc_hull_y_lower, spc_hull_y_upper,
    alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

pl.plot(x, spc_pp_y, 'k', color='#3F7F4C', marker='o', label=r"SPC$^{++}$")
pl.fill_between(x, spc_pp_y_lower, spc_pp_y_upper,
    alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99')
pl.ylabel("Number of Queries")
pl.xlabel("Instance Size")
pl.xticks([50,100,200])
pl.legend(loc='upper left')

pl.savefig("res/experiments_data/synthetic_plot.png", dpi=300)
pl.savefig("res/experiments_data/synthetic_plot_transparent.png", dpi=300, transparent = True)
pl.show()

s2 =  data[:, -1]
s2_y = np.ones(3)
s2_y[0] = np.average(s2[:50])
s2_y[1] = np.average(s2[50:100])
s2_y[2] = np.average(s2[100:])

s2_y_lower = np.ones(3)
s2_y_lower[0] = np.quantile(s2[:50], 0.05)
s2_y_lower[1] = np.quantile(s2[50:100], 0.05)
s2_y_lower[2] = np.quantile(s2[100:], 0.05)

s2_y_upper = np.ones(3)
s2_y_upper[0] = np.quantile(s2[:50], 0.95)
s2_y_upper[1] = np.quantile(s2[50:100], 0.95)
s2_y_upper[2] = np.quantile(s2[100:], 0.95)

pl.plot(x, s2_y, 'k', color='#e67e22', marker='o', label=r"S$^{2}$")
pl.fill_between(x, s2_y_lower, s2_y_upper,
    alpha=0.5, edgecolor='#e67e22', facecolor='#FFC300')

pl.ylabel("Accuracy")
pl.xlabel("Instance Size")
pl.xticks([50,100,200])
pl.legend(loc='upper left')

pl.savefig("res/experiments_data/s2.png", dpi=300, transparent = True)