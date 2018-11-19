import matplotlib.pyplot as plt

__all__ = ["show_curves"]

labels = ['awake', 'rem', 'stage1', 'stage2', 'stage3']
bands = ['delta', 'theta', 'alpha', 'beta']


def show_curves(data_50hz, data_10hz, target):
    sec_axis = 2 if len(data_50hz.size()) == 2 else data_50hz.size(1) + 1
    fig, axes = plt.subplots(max(data_50hz.size(0), data_10hz.size(0)), sec_axis, figsize=(20, 10))
    fig.suptitle(labels[target])
    for k in range(data_50hz.size(0)):
        if len(data_50hz.size()) == 3:  # pre processed into bands
            for i in range(data_50hz.size(1)):
                axes[k][i].plot(data_50hz[k, i].detach().cpu().numpy())
                axes[k][i].set_title(bands[i])
        else:
            axes[k][0].plot(data_50hz[k].detach().cpu().numpy())
    for k in range(data_10hz.size(0)):
        axes[k][-1].plot(data_10hz[k].detach().cpu().numpy())
