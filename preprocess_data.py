import pandas as pd
import numpy as np
import pickle as pkl
import os

def get_obs_min_max(in_file):
    df = pd.read_pickle(in_file)
    df = df[df['plev'] < 50000]
    df = df[df['plev'] > 30]
    y_min = df['obs_toz'].min()
    y_max = df['obs_toz'].max()
    return y_min, y_max

def get_obs(in_file):
    df = pd.read_pickle(in_file)
    df = df[df['plev'] < 50000]
    df = df[df['plev'] > 30]
    obs = df['obs_toz'].values
    return obs

def min_max_scale(y, a, b, y_min, y_max):
    return a + ((y - y_min) * (b - a) / (y_max - y_min))

def save_coords(in_file, save_dir):
    df = pd.read_pickle(in_file)
    df = df[df['plev'] < 50000]
    df = df[df['plev'] > 30]
    lats = np.unique(df['lat'].values)
    plevs = np.unique(df['plev'].values)
    time = dates = pd.date_range('19800101', freq='M', periods=372)
    pkl.dump(lats, open(os.path.join(save_dir, 'lats.pkl'), 'wb'))
    pkl.dump(plevs, open(os.path.join(save_dir, 'plevs.pkl'), 'wb'))
    pkl.dump(time, open(os.path.join(save_dir, 'dates.pkl'), 'wb'))
    return

def read_data(in_file, training=False):

    df = pd.read_pickle(in_file)
    df = df[df['plev'] < 50000]
    df = df[df['plev'] > 30]

    # Apply coordinate mapping lat -> z
    lat = df['lat'] 
    z = 2 * (df['lat'] - df['lat'].min())/(df['lat'].max() - df['lat'].min()) - 1

    # Apply coordinate mapping month_number -> x_mon, y_mon
    rads = (df['mon_num'] * 360/12) * (np.pi / 180)
    x_mon = np.sin(rads)
    y_mon = np.cos(rads)

    # min-max scale months (months since Jan 1980)
    mons_scaled = 2 * (df['mons'] - df['mons'].min())/(df['mons'].max() - df['mons'].min()) - 1

    # min-max scaling for levels - first log
    plev = np.log(df['plev'])
    lev_scaled = 2 * (plev - plev.min())/(plev.max() - plev.min()) - 1

    # Remove old coords and add new mapped coords from/to dataframe
    df = df.drop(['lat', 'plev', 'mon_num', 'mons'], axis=1)
    df['z'] = z
    df['lev'] = lev_scaled
    df['x_mon'] = x_mon
    df['y_mon'] = y_mon
    df['mons'] = mons_scaled

    # Apply log min-max scaling to each model and observations
    obs = df['obs_toz'].copy()
    # Mask concentrations less than 1ppb
    obs[np.log10(obs) < -9] = np.nan
    df['obs_toz'] = obs

    y_min = df['obs_toz'].min()
    y_max = df['obs_toz'].max()

    y_log_max = np.log(y_max)
    y_log_min = np.log(y_min)

    for i in range(14): # 13 models and 1 obs
        mdl = np.log(df[df.columns[i]])
        df[df.columns[i]] = min_max_scale(mdl, -1., 1., y_log_min, y_log_max)

    # Apply coordinate scaling
    df['z'] = df['z'] * 1.
    df['lev'] = df['lev'] * 1.
    df['x_mon'] = df['x_mon'] * 1.
    df['y_mon'] = df['y_mon'] * 1.
    df['mons'] = df['mons'] * 1.

    cols_to_drop = ['valid', 'temp_extrap', 'temp_interp', 'train', 'test']

    df_train = df[df['train']].drop(cols_to_drop, axis=1)
    df_test = df[df['test']].drop(cols_to_drop, axis=1)
    df_extrap = df[df['temp_extrap']].drop(cols_to_drop, axis=1)
    df_interp = df[df['temp_interp']].drop(cols_to_drop, axis=1)
    n_obs = len(df.dropna(inplace=False))

    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)
    df_interp.dropna(inplace=True)
    df_extrap.dropna(inplace=True)

    print('Number of obs: {}'.format(n_obs))
    print('Training on {:.1f}%'.format(100 * len(df_train)/n_obs))
    print('Testing on {:.1f}%'.format(100 * len(df_test)/n_obs))
    print('Validation (temporal extrapolation) on {:.1f}%'.format(100 * len(df_extrap)/n_obs))
    print('Validation (interpolation) on {:.1f}%'.format(100 * len(df_interp)/n_obs))

    # In sample training
    X_train = df_train.drop(['obs_toz'],axis=1).values.astype(np.float32)
    y_train = df_train['obs_toz'].values.reshape(-1,1).astype(np.float32)

    # The in sample testing - this is not used for training
    X_test = df_test.drop(['obs_toz'],axis=1).values.astype(np.float32)
    y_test = df_test['obs_toz'].values.reshape(-1,1).astype(np.float32)

    # For all time
    X_at = df.drop(['obs_toz'],axis=1).drop(cols_to_drop, axis=1).values.astype(np.float32)
    y_at = df.drop(cols_to_drop, axis=1)['obs_toz'].values.reshape(-1,1).astype(np.float32)

    # For interp
    X_interp = df_interp.drop(['obs_toz'],axis=1).values.astype(np.float32)
    y_interp = df_interp['obs_toz'].values.reshape(-1,1).astype(np.float32)

    # For extrap
    X_extrap = df_extrap.drop(['obs_toz'],axis=1).values.astype(np.float32)
    y_extrap = df_extrap['obs_toz'].values.reshape(-1,1).astype(np.float32)


    if training:
        return X_train, y_train
    else:
        return X_train, y_train, X_test, y_test, X_interp, y_interp, X_extrap, y_extrap, X_at, y_at
