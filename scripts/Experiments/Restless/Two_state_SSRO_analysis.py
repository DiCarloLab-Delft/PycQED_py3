a=ma.MeasurementAnalysis(auto=False)
a.get_naming_and_values()
I_shots = a.measured_values[0]
Q_shots = a.measured_values[1]

I_0 = I_shots[::3]
I_1 = I_shots[1::3]
I_2 = I_shots[2::3]

Q_0 = Q_shots[::3]
Q_1 = Q_shots[1::3]
Q_2 = Q_shots[2::3]


f,axs= plt.subplots(1,3)
f.set_figwidth(16)
axs[0].hist2d(I_0, Q_0, cmap='viridis', bins=40)
axs[1].hist2d(I_1, Q_1, cmap='viridis', bins=40)
axs[2].hist2d(I_2, Q_2, cmap='viridis', bins=40)
for ax in axs:
    ax.set_xlim(min(min(I_0), min(I_1), min(I_2)), max(max(I_0), max(I_1), max(I_2)))
    ax.set_ylim(min(min(Q_0), min(Q_1), min(Q_2)), max(max(Q_0), max(Q_1), max(Q_2)))
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.plot(np.mean(I_0), np.mean(Q_0), marker = 'o', markersize=15, c='b', alpha=0.9)
    ax.plot(np.mean(I_1), np.mean(Q_1), marker = 'o', markersize=15, c='r', alpha=0.9)
    ax.plot(np.mean(I_2), np.mean(Q_2), marker = 'o', markersize=15, c='g', alpha=0.9)