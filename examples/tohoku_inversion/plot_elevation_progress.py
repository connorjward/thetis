import h5py
import matplotlib.pyplot as plt


station_names = [
    '801', '802', '803', '804', '806', '807',
    'P02', 'P06', 'KPG1', 'KPG2', 'MPG1', 'MPG2',
    '21401', '21413', '21418', '21419',
]

nplots = len(station_names)

fig = plt.figure(figsize=(40, 20))
axes = fig.subplots(4, 4)

inv_dir = 'outputs_elev-init-optimization'

for i, sta in enumerate(station_names):

    o = f'observations/diagnostic_timeseries_{sta}_elev.hdf5'
    with h5py.File(o, 'r') as h5file:
        time_obs = h5file['time'][:].flatten()/60.0
        vals_obs = h5file['elev'][:].flatten()

    f = f'{inv_dir}/diagnostic_timeseries_progress_{sta}_elev.hdf5'
    with h5py.File(f, 'r') as h5file:
        time = h5file['time'][:].flatten()/60.0
        vals = h5file['elev'][:]

    ax = axes[i//4, i % 4]
    ax.plot(time_obs, vals_obs, 'k', zorder=3, label='Observation', lw=1.3)
    for i, v in enumerate(vals):
        if sta[:2] in ['80', 'P0']:
            v -= v[0]
        ax.plot(time, v, label=f'_iter {i}', lw=0.5)
    ax.set_title(sta)
    ax.set_xlim([time_obs[0], time_obs[-1]])
    if i // 4 == 3:
        ax.set_xlabel('Time (min)')
    if i % 4 == 0:
        ax.set_ylabel('Elevation (m)')
    ax.legend(ncol=1)
    ax.grid(True)

imgfile = 'optimization_progress_elev_ts.png'
print(f'Saving to {imgfile}')
plt.savefig(imgfile, dpi=200, bbox_inches='tight')
