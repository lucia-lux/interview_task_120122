import os
import RCT_analysis
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from RCT_analysis import RCT_utilities as rctu

# load data
in_df = pd.read_csv(os.path.join(os.getcwd(), "data.csv"))
# Preprocess
pp = rctu.Preprocess_Dataframe(in_df)
pp.prepare_df()
# Get baseline description
de = rctu.Data_Explorer(pp.in_df_long,'Group','time')
de.describe_sample(['GHQ','SWLS','PSS','RWS','Age'],['Gender'])
# visualize the data (save to file)
dv = rctu.Data_Visualizer(pp.in_df_long,'Group','time')
# names of outcome variables
vars = ['SWLS','PSS','RWS']
# get group names
groups = pp.in_df_long.Group.unique()
# Visualize
for var in vars:
    fig,ax = plt.subplots(len(groups),1,figsize = (20,15))
    for num,group_name in enumerate(groups):
        dv.draw_kdeplot(ax[num], group_name,var)
        ax[num].title.set_text(group_name)
    fig.savefig(''.join([var,".png"]))
    # analyze
    da = rctu.Data_Analyzer(pp.in_df_long,'Group','time','id',var)
    da.remove_excess_missing()
    da.remove_excess_missing(criterion = [1,2])
    print("\nBuilding model...")
    #da.build_lmem(var,["C(Group,Treatment('B'))", "C(time,Treatment(0))"],interacts = 1)
    #da.model_diagnostics('lmem')
    da.build_gee(var,["C(Group,Treatment('B'))", "C(time,Treatment(0))"],sm.families.Gaussian(), sm.cov_struct.Autoregressive(),interacts = 1)
    da.model_diagnostics('gee')